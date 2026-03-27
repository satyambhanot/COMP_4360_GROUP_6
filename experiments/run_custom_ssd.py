# experiments/run_custom_ssd.py
# trains and evaluates our custom SSD with MobileNetV2 backbone
# saves everything to saved_results/custom_ssd/
#
# usage: python experiments/run_custom_ssd.py

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import config
config.MODEL_TYPE = "custom_ssd"
config.BATCH_SIZE = 16
config.NUM_EPOCHS = 150
config.EARLY_STOPPING_PATIENCE = 25

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saved_results", "custom_ssd")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 50)
    print("  Experiment: Custom SSD (MobileNetV2)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 50)
    print()

    print("--- Training ---")
    from train import run_training
    model, best_val_loss, history = run_training(checkpoint_dir=CHECKPOINT_DIR)

    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Evaluation ---")
    from evaluate import run_evaluation
    metrics = run_evaluation(output_dir=PLOTS_DIR, checkpoint_dir=CHECKPOINT_DIR)

    import shutil
    if os.path.exists(os.path.join(PLOTS_DIR, "evaluation_results.json")):
        shutil.copy(
            os.path.join(PLOTS_DIR, "evaluation_results.json"),
            os.path.join(OUTPUT_DIR, "metrics.json")
        )

    print(f"\n{'=' * 50}")
    print(f"  Custom SSD experiment complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
