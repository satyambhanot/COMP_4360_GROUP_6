# experiments/run_ablation_variant.py
# trains and evaluates a single ablation variant
# called by run_ablation.py with command line arguments
# runs as a separate process so GPU memory is fully released between variants
#
# usage: python experiments/run_ablation_variant.py <variant_name> <use_hsv> <use_sobel>
# example: python experiments/run_ablation_variant.py full_pipeline True True

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# override config for RetinaNet with reduced epochs for ablation
import config
config.MODEL_TYPE = "retinanet"
config.BATCH_SIZE = 8
config.NUM_EPOCHS = 30
config.EARLY_STOPPING_PATIENCE = 15

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "saved_results", "ablation")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python run_ablation_variant.py <name> <use_hsv> <use_sobel>")
        print("example: python run_ablation_variant.py full_pipeline True True")
        exit(1)

    variant_name = sys.argv[1]
    use_hsv = sys.argv[2].lower() == "true"
    use_sobel = sys.argv[3].lower() == "true"

    variant_dir = os.path.join(OUTPUT_BASE, variant_name)
    checkpoint_dir = os.path.join(variant_dir, "checkpoints")
    plots_dir = os.path.join(variant_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"{'=' * 50}")
    print(f"  Ablation Variant: {variant_name}")
    print(f"  HSV: {use_hsv}, Sobel: {use_sobel}")
    print(f"  Output: {variant_dir}")
    print(f"{'=' * 50}", flush=True)

    # create model with specific preprocessing flags
    from model import create_model
    model = create_model(use_hsv=use_hsv, use_sobel=use_sobel)

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
            "use_hsv": use_hsv,
            "use_sobel": use_sobel,
            "metrics": metrics
        }
        with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n  {variant_name} complete!", flush=True)
