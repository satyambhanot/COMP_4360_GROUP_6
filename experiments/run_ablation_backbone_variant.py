'''
run_ablation_backbone_variant.py
trains and evaluates a single backbone initialization variant
called by run_ablation_backbone.py as a separate process for clean GPU memory

usage: python experiments/run_ablation_backbone_variant.py <name> <use_pretrained>
'''

import sys
import os
import json

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

'''
override config for custom SSD with reduced epochs for ablation
must happen before importing train/evaluate so they pick up the changes
'''
import config
config.MODEL_TYPE = "custom_ssd"
config.BATCH_SIZE = 16
config.NUM_EPOCHS = 30
config.EARLY_STOPPING_PATIENCE = 15

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "saved_results", "ablation_backbone")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python run_ablation_backbone_variant.py <name> <use_pretrained>")
        print("example: python run_ablation_backbone_variant.py pretrained True")
        exit(1)

    variant_name = sys.argv[1]
    use_pretrained = sys.argv[2].lower() == "true"

    variant_dir = os.path.join(OUTPUT_BASE, variant_name)
    checkpoint_dir = os.path.join(variant_dir, "checkpoints")
    plots_dir = os.path.join(variant_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"{'=' * 50}")
    print(f"  Ablation Variant: {variant_name}")
    print(f"  Pretrained: {use_pretrained}")
    print(f"  Output: {variant_dir}")
    print(f"{'=' * 50}", flush=True)

    # create custom SSD with pretrained or random backbone
    from models.model import create_model
    model = create_model(use_pretrained_backbone=use_pretrained)

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
            "use_pretrained": use_pretrained,
            "metrics": metrics
        }
        with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n  {variant_name} complete!", flush=True)
