# experiments/run_yolo.py
# trains and evaluates YOLOv8 for comparison against our SSD/RetinaNet approach
# saves everything to saved_results/yolo/
#
# usage: python experiments/run_yolo.py

import sys
import os
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saved_results", "yolo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=" * 50)
    print("  Experiment: YOLOv8 (Comparison)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 50)
    print()

    # run the existing train_yolo.py which handles everything
    from train_yolo import prepare_yolo_dataset, train_yolo, evaluate_yolo

    yaml_path = prepare_yolo_dataset()
    model = train_yolo(yaml_path)

    if model is not None:
        results = evaluate_yolo(model)

        # save to our output directory
        if results:
            with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
                json.dump(results, f, indent=2)

        # copy any training plots that ultralytics generated
        yolo_train_dir = os.path.join(PROJECT_ROOT, "results", "yolo", "train")
        if os.path.exists(yolo_train_dir):
            plots_dir = os.path.join(OUTPUT_DIR, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            for f in os.listdir(yolo_train_dir):
                if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".csv"):
                    shutil.copy(os.path.join(yolo_train_dir, f), plots_dir)

    print(f"\n{'=' * 50}")
    print(f"  YOLOv8 experiment complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
