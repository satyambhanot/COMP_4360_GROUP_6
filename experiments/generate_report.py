# experiments/generate_report.py
# scans saved_results/ for all completed experiments and generates
# comparison charts, summary tables, and a combined results file
#
# this is the final step after all experiments are done
# it reads metrics.json from each model folder and creates
# side by side comparisons for the report
#
# usage: python experiments/generate_report.py

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAVED_RESULTS = os.path.join(PROJECT_ROOT, "saved_results")
REPORT_DIR = os.path.join(SAVED_RESULTS, "report")


def find_all_results():
    # scans saved_results/ for all model folders that have metrics.json
    # returns a dict of model_name -> metrics

    models = {}

    # check each subdirectory for a metrics.json file
    model_dirs = {
        "Custom SSD": "custom_ssd",
        "Pretrained SSD300": "pretrained_ssd",
        "RetinaNet": "retinanet",
        "YOLOv8": "yolo"
    }

    for display_name, folder_name in model_dirs.items():
        metrics_path = os.path.join(SAVED_RESULTS, folder_name, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)

            # normalize the metrics format since different models save differently
            if "metrics" in data and isinstance(data["metrics"], dict):
                metrics = data["metrics"]
            else:
                metrics = data

            models[display_name] = metrics
            print(f"  found {display_name}: F1={metrics.get('f1', 0):.4f}")
        else:
            print(f"  {display_name}: no results found")

    return models


def find_ablation_results():
    # looks for ablation results in saved_results/ablation/comparison/
    ablation_path = os.path.join(SAVED_RESULTS, "ablation", "comparison", "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            return json.load(f)
    return None


def plot_model_comparison(models):
    # grouped bar chart comparing precision, recall, F1 across all models

    if len(models) < 2:
        print("need at least 2 models to compare")
        return

    model_names = list(models.keys())
    metric_keys = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1 Score"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 3), 7))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        values = [models[n].get(key, 0) for n in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Architecture Comparison: Where's Waldo Detection", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"  saved model_comparison.png")


def plot_improvement_journey(models):
    # line chart showing F1 progression across model iterations

    order = ["Custom SSD", "Pretrained SSD300", "RetinaNet", "YOLOv8"]
    ordered = [(n, models[n]) for n in order if n in models]

    if len(ordered) < 2:
        print("need at least 2 models for journey chart")
        return

    names = [n for n, _ in ordered]
    f1_scores = [m.get("f1", 0) for _, m in ordered]
    precision = [m.get("precision", 0) for _, m in ordered]
    recall = [m.get("recall", 0) for _, m in ordered]

    x = range(len(names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, f1_scores, marker="o", markersize=12, linewidth=3, label="F1 Score", color="#2196F3")
    ax.plot(x, precision, marker="s", markersize=8, linewidth=2, label="Precision", color="#4CAF50", alpha=0.7)
    ax.plot(x, recall, marker="^", markersize=8, linewidth=2, label="Recall", color="#FF9800", alpha=0.7)

    for i, (name, f1) in enumerate(zip(names, f1_scores)):
        ax.annotate(f"{f1:.3f}", (i, f1), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Improvement Journey", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "improvement_journey.png"), dpi=150)
    plt.close()
    print(f"  saved improvement_journey.png")


def generate_summary_table(models):
    # clean comparison table as image and text

    model_names = list(models.keys())
    headers = ["Model", "Precision", "Recall", "F1", "Accuracy"]

    rows = []
    for name in model_names:
        m = models[name]
        rows.append([
            name,
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
            f"{m.get('accuracy', 0):.4f}" if m.get('accuracy', 0) > 0 else "n/a"
        ])

    # text version
    with open(os.path.join(REPORT_DIR, "summary_table.txt"), "w") as f:
        f.write(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}\n")
        f.write("-" * 70 + "\n")
        for row in rows:
            f.write(f"{row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}\n")
    print(f"  saved summary_table.txt")

    # image version
    fig, ax = plt.subplots(figsize=(12, max(3, 0.6 * len(rows) + 2)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # highlight best F1
    best_idx = 0
    best_f1 = 0
    for i, row in enumerate(rows):
        try:
            f1 = float(row[3])
            if f1 > best_f1:
                best_f1 = f1
                best_idx = i
        except ValueError:
            pass
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor("#C8E6C9")

    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved summary_table.png")


def save_combined_results(models, ablation):
    # saves everything to one JSON for easy reference

    combined = {
        "model_comparison": {},
        "ablation": ablation
    }

    for name, metrics in models.items():
        combined["model_comparison"][name] = metrics

    with open(os.path.join(REPORT_DIR, "all_results.json"), "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  saved all_results.json")


if __name__ == "__main__":
    print("=" * 50)
    print("  Report Generator")
    print(f"  Scanning: {SAVED_RESULTS}")
    print("=" * 50)
    print()

    os.makedirs(REPORT_DIR, exist_ok=True)

    # find all model results
    print("looking for model results...")
    models = find_all_results()

    if len(models) == 0:
        print("\nno results found. run some experiments first:")
        print("  python experiments/run_retinanet.py")
        print("  python experiments/run_pretrained_ssd.py")
        print("  python experiments/run_yolo.py")
        exit()

    print(f"\nfound {len(models)} models")

    # find ablation results
    ablation = find_ablation_results()
    if ablation:
        print(f"found ablation results with {len(ablation)} variants")

    # generate all outputs
    print("\ngenerating charts...")
    plot_model_comparison(models)
    plot_improvement_journey(models)
    generate_summary_table(models)
    save_combined_results(models, ablation)

    print(f"\n{'=' * 50}")
    print(f"  Report generated!")
    print(f"  Check: {REPORT_DIR}")
    print(f"{'=' * 50}")
