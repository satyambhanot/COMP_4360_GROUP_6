'''
generate_report.py
scans saved_results/ for all completed experiments and generates
comparison charts, summary tables, and combined results file
this is the final step after all experiments are done

covers:
  model comparison (custom SSD, pretrained SSD, RetinaNet, YOLOv8)
  ablation 1: preprocessing (full pipeline, HSV only, RGB only)
  ablation 2: SSD head count (5, 4, 3 heads)
  ablation 3: tile resolution (64, 128, 256)
  ablation 4: color information (RGB, grayscale, binary)
  ablation 5: pretrained vs random backbone

usage: python experiments/generate_report.py
'''

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
    '''scans saved_results/ for metrics.json in each model folder'''

    models = {}

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

            if "metrics" in data and isinstance(data["metrics"], dict):
                metrics = data["metrics"]
            else:
                metrics = data

            models[display_name] = metrics
            print(f"  found {display_name}: F1={metrics.get('f1', 0):.4f}")
        else:
            print(f"  {display_name}: no results found")

    return models


def find_ablation_results(ablation_dir, variant_names):
    '''
    reads ablation results from a specific ablation directory
    generic function that works for any ablation study

    args:
        ablation_dir: path to the ablation folder (eg saved_results/ablation)
        variant_names: list of variant folder names to look for

    returns:
        list of dicts with name and metrics, or None if nothing found
    '''

    if not os.path.exists(ablation_dir):
        return None

    results = []
    for variant in variant_names:
        metrics_path = os.path.join(ablation_dir, variant, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            if "metrics" in data and isinstance(data["metrics"], dict):
                metrics = data["metrics"]
            else:
                metrics = data
            results.append({"name": variant, "metrics": metrics})

    return results if len(results) > 0 else None


def find_all_ablations():
    '''
    scans saved_results/ for all 5 ablation studies
    returns a dict mapping ablation name to its results list
    '''

    ablations = {}

    '''
    each entry: (display name, folder name, list of variant folder names)
    the variant names must match what the ablation scripts create
    '''
    ablation_configs = [
        ("Preprocessing", "ablation", ["full_pipeline", "hsv_only", "rgb_only"]),
        ("SSD Heads", "ablation_heads", ["5_heads", "4_heads", "3_heads"]),
        ("Tile Resolution", "ablation_tiles", ["tiles_128", "tiles_256"]),
        ("Color Information", "ablation_color", ["color_rgb", "color_gray", "color_binary"]),
        ("Backbone Init", "ablation_backbone", ["pretrained", "random_init"]),
    ]

    for display_name, folder_name, variant_names in ablation_configs:
        ablation_dir = os.path.join(SAVED_RESULTS, folder_name)
        results = find_ablation_results(ablation_dir, variant_names)
        if results:
            ablations[display_name] = results
            variant_count = len(results)
            print(f"  found {display_name}: {variant_count} variants")
            for r in results:
                m = r["metrics"]
                print(f"    {r['name']}: F1={m.get('f1', 0):.4f}")

    return ablations


def plot_model_comparison(models):
    '''creates grouped bar chart comparing precision, recall, F1 across all models'''

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
    '''creates line chart showing F1 progression across model iterations'''

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


def plot_ablation_comparison(ablation_name, results, filename):
    '''
    creates a grouped bar chart for one ablation study
    reusable for all 5 ablation types

    args:
        ablation_name: display title for the chart
        results: list of dicts with name and metrics
        filename: output PNG filename (without path)
    '''

    if len(results) < 2:
        print(f"  need at least 2 variants for {ablation_name}")
        return

    '''
    display name mappings for each ablation study
    maps internal variant names to readable labels for the chart
    '''
    display_maps = {
        "Preprocessing": {
            "full_pipeline": "RGB+HSV+Sobel",
            "hsv_only": "RGB+HSV",
            "rgb_only": "RGB Only"
        },
        "SSD Heads": {
            "5_heads": "5 Heads",
            "4_heads": "4 Heads",
            "3_heads": "3 Heads"
        },
        "Tile Resolution": {
            "tiles_128": "128x128",
            "tiles_256": "256x256"
        },
        "Color Information": {
            "color_rgb": "RGB Color",
            "color_gray": "Grayscale",
            "color_binary": "Binary B/W"
        },
        "Backbone Init": {
            "pretrained": "Pretrained",
            "random_init": "Random Init"
        },
    }

    display = display_maps.get(ablation_name, {})
    names = [r["name"] for r in results]
    labels = [display.get(n, n) for n in names]

    metric_keys = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 3), 6))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        values = [r["metrics"].get(key, 0) for r in results]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Study: {ablation_name}", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150)
    plt.close()
    print(f"  saved {filename}")


def generate_summary_table(models):
    '''creates a comparison table as both image and text file'''

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


def generate_ablation_summary_table(ablations):
    '''
    creates a combined text summary table for all ablation studies
    each ablation gets its own section with variant results
    '''

    with open(os.path.join(REPORT_DIR, "ablation_summary.txt"), "w") as f:
        for ablation_name, results in ablations.items():
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  {ablation_name}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"{'Variant':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
            f.write("-" * 60 + "\n")
            for r in results:
                m = r["metrics"]
                f.write(f"{r['name']:<25} {m.get('precision', 0):>10.4f} "
                        f"{m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f}\n")

    print(f"  saved ablation_summary.txt")


def save_combined_results(models, ablations):
    '''saves all model and ablation results to a single JSON file'''

    combined = {
        "model_comparison": {},
        "ablations": {}
    }

    for name, metrics in models.items():
        combined["model_comparison"][name] = metrics

    for ablation_name, results in ablations.items():
        combined["ablations"][ablation_name] = results

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

    # find all ablation results
    print("\nlooking for ablation results...")
    ablations = find_all_ablations()
    print(f"found {len(ablations)} ablation studies")

    # generate model comparison charts
    print("\ngenerating model charts...")
    plot_model_comparison(models)
    plot_improvement_journey(models)
    generate_summary_table(models)

    '''
    generate ablation charts
    each ablation gets its own comparison bar chart in the report folder
    the filename maps ensure consistent naming across runs
    '''
    ablation_filenames = {
        "Preprocessing": "ablation_preprocessing.png",
        "SSD Heads": "ablation_heads.png",
        "Tile Resolution": "ablation_tiles.png",
        "Color Information": "ablation_color.png",
        "Backbone Init": "ablation_backbone.png",
    }

    if ablations:
        print("\ngenerating ablation charts...")
        for ablation_name, results in ablations.items():
            filename = ablation_filenames.get(ablation_name, f"ablation_{ablation_name.lower()}.png")
            plot_ablation_comparison(ablation_name, results, filename)
        generate_ablation_summary_table(ablations)

    # save everything to one JSON
    save_combined_results(models, ablations)

    print(f"\n{'=' * 50}")
    print(f"  Report generated!")
    print(f"  Check: {REPORT_DIR}")
    print(f"{'=' * 50}")
