'''
run_ablation_tiles_variant.py
trains and evaluates a single tile resolution variant
called by run_ablation_tiles.py as a separate process for clean GPU memory

each variant re-tiles the entire dataset at the specified tile size
using a separate processed directory so different tile sizes dont conflict
tile offsets are scaled to half the tile size for consistent overlap strategy

usage: python experiments/run_ablation_tiles_variant.py <name> <tile_size>
'''

import sys
import os
import json

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

'''
override config for RetinaNet with reduced epochs for ablation
must happen before importing train/evaluate so they pick up the changes
the tile size and offsets are overridden dynamically below
'''
import config
config.MODEL_TYPE = "retinanet"
config.BATCH_SIZE = 8
config.NUM_EPOCHS = 30
config.EARLY_STOPPING_PATIENCE = 15

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "saved_results", "ablation_tiles")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python run_ablation_tiles_variant.py <name> <tile_size>")
        print("example: python run_ablation_tiles_variant.py tiles_128 128")
        exit(1)

    variant_name = sys.argv[1]
    tile_size = int(sys.argv[2])

    variant_dir = os.path.join(OUTPUT_BASE, variant_name)
    checkpoint_dir = os.path.join(variant_dir, "checkpoints")
    plots_dir = os.path.join(variant_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    '''
    override tile configuration before the data pipeline runs
    each tile size gets its own processed directory so they dont overwrite each other
    tile offsets are half the tile size to match the overlap strategy used for 256x256
    '''
    half = tile_size // 2
    config.TILE_SIZES = [tile_size]
    config.TILE_OFFSETS = [(0, 0), (half, 0), (0, half), (half, half)]

    '''
    use a separate processed directory for each tile size
    this prevents different variants from overwriting each others tiles and splits
    '''
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed_ablation_tiles", str(tile_size))
    config.PROCESSED = processed_dir
    os.makedirs(processed_dir, exist_ok=True)

    print(f"{'=' * 50}")
    print(f"  Ablation Variant: {variant_name}")
    print(f"  Tile Size: {tile_size}x{tile_size}")
    print(f"  Tile Offsets: {config.TILE_OFFSETS}")
    print(f"  Processed Dir: {processed_dir}")
    print(f"  Output: {variant_dir}")
    print(f"{'=' * 50}", flush=True)

    '''
    re-run the data pipeline to tile images at this resolution
    this creates new tiles and splits.json in the variant-specific processed directory
    '''
    print("\n--- Tiling Data ---", flush=True)
    from data.data_pipeline import run_pipeline
    run_pipeline()

    # create model
    from models.model import create_model
    model = create_model()

    # train
    print("\n--- Training ---", flush=True)
    from train import run_training
    model, val_loss, history = run_training(
        checkpoint_dir=checkpoint_dir,
        model_override=model
    )

    # free GPU
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # evaluate
    print("\n--- Evaluation ---", flush=True)
    from evaluate import run_evaluation
    metrics = run_evaluation(output_dir=plots_dir, checkpoint_dir=checkpoint_dir)

    # save metrics
    if metrics:
        result = {
            "variant": variant_name,
            "tile_size": tile_size,
            "metrics": metrics
        }
        with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    '''
    clean up tiled data to free disk space
    smaller tile sizes produce vastly more tiles (64x64 creates ~16x more than 256x256)
    which can fill the disk before the next variant starts
    the tiles are no longer needed after training and evaluation are done
    checkpoints and metrics are saved separately in the variant folder
    '''
    import shutil
    if os.path.exists(processed_dir):
        print(f"\n  cleaning up tile data at {processed_dir}...", flush=True)
        shutil.rmtree(processed_dir)
        print(f"  freed disk space", flush=True)

    print(f"\n  {variant_name} complete!", flush=True)
