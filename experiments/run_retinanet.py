# experiments/run_retinanet.py
# trains and evaluates the RetinaNet model
# saves everything to saved_results/retinanet/
# completely self contained, no manual config editing needed
#
# usage: python experiments/run_retinanet.py

import sys
import os

# add project root to path so we can import our modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# override config BEFORE importing anything that reads it
# this way the model type is set correctly when train.py and evaluate.py load
import config
config.MODEL_TYPE = "retinanet"
config.BATCH_SIZE = 8
config.RETINANET_INPUT_SIZE = 512
config.NUM_EPOCHS = 150
config.EARLY_STOPPING_PATIENCE = 25

# output directory for this experiment
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saved_results", "retinanet")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 50)
    print("  Experiment: RetinaNet")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 50)
    print()

    # step 1: train
    print("--- Training ---")
    from train import run_training
    model, best_val_loss, history = run_training(checkpoint_dir=CHECKPOINT_DIR)

    # free GPU memory before evaluation
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # step 2: evaluate
    print("\n--- Evaluation ---")
    from evaluate import run_evaluation
    metrics = run_evaluation(output_dir=PLOTS_DIR, checkpoint_dir=CHECKPOINT_DIR)

    # step 3: copy the main metrics json to the top level of the output dir
    import shutil
    if os.path.exists(os.path.join(PLOTS_DIR, "evaluation_results.json")):
        shutil.copy(
            os.path.join(PLOTS_DIR, "evaluation_results.json"),
            os.path.join(OUTPUT_DIR, "metrics.json")
        )

    print(f"\n{'=' * 50}")
    print(f"  RetinaNet experiment complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
