'''
run_ablation.py
controller for the preprocessing ablation study
runs each variant as a separate process so GPU memory is fully released

tests whether HSV mask and Sobel edge detection actually help:
  full_pipeline: RGB + HSV + Sobel
  hsv_only: RGB + HSV (no Sobel)
  rgb_only: just RGB (no preprocessing)

usage: python experiments/run_ablation.py
'''

import sys
import os
import json
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saved_results", "ablation")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparison")

# the three variants we test
# each one gets its own python process for clean GPU memory
VARIANTS = [
    ("full_pipeline", "True", "True"),    # RGB + HSV + Sobel
    ("hsv_only", "True", "False"),         # RGB + HSV only
    ("rgb_only", "False", "False"),        # just RGB
]


def run_variant_process(name, use_hsv, use_sobel):
    '''launches one ablation variant as a separate process for clean GPU memory'''

    script = os.path.join(PROJECT_ROOT, "experiments", "run_ablation_variant.py")
    cmd = [sys.executable, script, name, use_hsv, use_sobel]

    print(f"\nlaunching: {' '.join(cmd)}", flush=True)
    start = time.time()

    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})

    elapsed = (time.time() - start) / 60
    if result.returncode == 0:
        print(f"  {name} completed in {elapsed:.0f} minutes", flush=True)
    else:
        print(f"  {name} FAILED with return code {result.returncode}", flush=True)

    return result.returncode == 0


def generate_comparison():
    '''loads metrics from all variants and creates a comparison bar chart'''

    os.makedirs(COMPARISON_DIR, exist_ok=True)

    all_results = []
    for name, _, _ in VARIANTS:
        metrics_path = os.path.join(OUTPUT_DIR, name, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            all_results.append({
                "name": name,
                "metrics": data.get("metrics", data)
            })
            m = data.get("metrics", data)
            print(f"  {name}: P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f} F1={m.get('f1', 0):.4f}")
        else:
            print(f"  {name}: no results found")

    if len(all_results) < 2:
        print("need at least 2 variants to compare")
        return

    # create comparison bar chart
    names = [r["name"] for r in all_results]
    display = {"full_pipeline": "RGB + HSV + Sobel", "hsv_only": "RGB + HSV", "rgb_only": "RGB Only"}
    labels = [display.get(n, n) for n in names]

    metric_keys = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        values = [r["metrics"].get(key, 0) for r in all_results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Preprocessing Impact on RetinaNet")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "preprocessing_comparison.png"), dpi=150)
    plt.close()
    print(f"  saved comparison chart to {COMPARISON_DIR}/preprocessing_comparison.png")

    # save combined results JSON
    with open(os.path.join(COMPARISON_DIR, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  saved combined results to {COMPARISON_DIR}/ablation_results.json")


if __name__ == "__main__":
    print("=" * 50)
    print("  Ablation Study: Preprocessing Impact")
    print("  Each variant runs as a separate process")
    print("  so GPU memory is fully released between them")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_start = time.time()

    '''
    run each variant as a separate process
    sleep 5 seconds between variants to give the system time to fully
    release GPU memory from the previous process before the next one starts
    '''
    for i, (name, use_hsv, use_sobel) in enumerate(VARIANTS):
        if i > 0:
            print("\n  waiting 5s for GPU memory to fully release...", flush=True)
            time.sleep(5)
        success = run_variant_process(name, use_hsv, use_sobel)
        if not success:
            print(f"\n  {name} failed, stopping ablation study")
            break

    # generate comparison chart from whatever results we have
    print("\n--- Generating Comparison ---", flush=True)
    generate_comparison()

    total_time = (time.time() - total_start) / 60
    print(f"\n{'=' * 50}")
    print(f"  Ablation study complete! ({total_time:.0f} minutes)")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"{'=' * 50}")
