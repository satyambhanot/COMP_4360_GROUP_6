# compare_results.py
# generates comparison charts across all model architectures we tested
# this is for the report section where we show the progression of improvements
# and compare our approach against different baselines
#
# it loads results from:
#   results/evaluation_results.json (whichever model was last trained with run.sh)
#   results/yolo/yolo_results.json (from train_yolo.py)
#   results/ssd_results.json (manually saved from a pretrained SSD run)
#   results/retinanet_results.json (manually saved from a RetinaNet run)
#
# to use this properly you need to:
#   1. train with MODEL_TYPE = "pretrained_ssd", copy evaluation_results.json to ssd_results.json
#   2. train with MODEL_TYPE = "retinanet", copy evaluation_results.json to retinanet_results.json
#   3. run train_yolo.py to get yolo results
#   4. run this script to generate comparison charts
#
# usage: python compare_results.py

import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")


def load_all_results():
    # tries to load results from each model type
    # returns a dict of model_name -> metrics
    # any model that doesnt have results yet gets skipped

    models = {}

    # check for saved results from different runs
    # these need to be manually copied after each training run
    # because evaluation_results.json gets overwritten each time
    result_files = {
        "Custom SSD": os.path.join(RESULTS_DIR, "custom_ssd_results.json"),
        "Pretrained SSD300": os.path.join(RESULTS_DIR, "ssd_results.json"),
        "RetinaNet": os.path.join(RESULTS_DIR, "retinanet_results.json"),
    }

    for name, path in result_files.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            models[name] = data["metrics"]
            print(f"  loaded {name} results from {path}")

    # also check if the current evaluation_results.json exists
    # and hasnt been copied to a model specific file yet
    current_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(current_path):
        with open(current_path) as f:
            data = json.load(f)
        # only add if we dont already have results for this model type
        # we dont know which model generated it so label it as "Latest Run"
        if "Latest Run" not in models:
            models["Latest Run"] = data["metrics"]
            print(f"  loaded latest run from {current_path}")

    # check for YOLO results
    yolo_path = os.path.join(RESULTS_DIR, "yolo", "yolo_results.json")
    if os.path.exists(yolo_path):
        with open(yolo_path) as f:
            data = json.load(f)
        models["YOLOv8"] = data["metrics"]
        print(f"  loaded YOLOv8 results from {yolo_path}")

    return models


def plot_model_comparison(models):
    # creates a grouped bar chart comparing precision, recall, and F1
    # across all models we tested. this is the main comparison chart
    # for the report, showing at a glance which model works best

    if len(models) < 2:
        print("need at least 2 models to compare, skipping comparison chart")
        return

    model_names = list(models.keys())
    metrics_to_plot = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1 Score"]

    x = np.arange(len(model_names))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 6))

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        values = []
        for name in model_names:
            m = models[name]
            values.append(m.get(metric, 0))
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        # put the value on top of each bar
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Model Architecture Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"saved model comparison chart to results/comparison/model_comparison.png")


def plot_improvement_journey(models):
    # shows the F1 score progression across our model iterations
    # this tells the story of how we improved step by step
    # the professor will appreciate seeing the thought process

    if len(models) < 2:
        print("need at least 2 models for journey chart, skipping")
        return

    # order the models in the progression we tried them
    # custom SSD first, then pretrained SSD, then RetinaNet, then YOLO
    order = ["Custom SSD", "Pretrained SSD300", "RetinaNet", "YOLOv8", "Latest Run"]
    ordered_names = [n for n in order if n in models]

    # if we dont have the ordered names, just use what we have
    if len(ordered_names) < 2:
        ordered_names = list(models.keys())

    f1_scores = [models[n].get("f1", 0) for n in ordered_names]
    precision_scores = [models[n].get("precision", 0) for n in ordered_names]
    recall_scores = [models[n].get("recall", 0) for n in ordered_names]

    x = range(len(ordered_names))

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_names) * 2), 6))
    ax.plot(x, f1_scores, marker="o", markersize=10, linewidth=2.5, label="F1 Score", color="#2196F3")
    ax.plot(x, precision_scores, marker="s", markersize=8, linewidth=1.5, label="Precision", color="#4CAF50", alpha=0.7)
    ax.plot(x, recall_scores, marker="^", markersize=8, linewidth=1.5, label="Recall", color="#FF9800", alpha=0.7)

    # annotate each point with its F1 value
    for i, (name, f1) in enumerate(zip(ordered_names, f1_scores)):
        ax.annotate(f"{f1:.3f}", (i, f1), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Improvement Journey")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "improvement_journey.png"), dpi=150)
    plt.close()
    print(f"saved improvement journey to results/comparison/improvement_journey.png")


def generate_summary_table(models):
    # saves a clean summary table as both an image and a text file
    # the image can go directly into the report
    # the text file is useful for quick reference

    if len(models) == 0:
        print("no models to summarize")
        return

    model_names = list(models.keys())
    metric_names = ["precision", "recall", "f1", "accuracy"]
    headers = ["Model"] + [m.title() for m in metric_names]

    rows = []
    for name in model_names:
        m = models[name]
        row = [name]
        for metric in metric_names:
            val = m.get(metric, 0)
            row.append(f"{val:.4f}" if val > 0 else "n/a")
        rows.append(row)

    # save as text
    text_path = os.path.join(COMPARISON_DIR, "summary_table.txt")
    with open(text_path, "w") as f:
        # header
        f.write(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}\n")
        f.write("-" * 70 + "\n")
        for row in rows:
            f.write(f"{row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}\n")
    print(f"saved summary table to results/comparison/summary_table.txt")

    # save as image
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # style the header row
    for j in range(len(headers)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # highlight the best F1 row in green
    best_f1_idx = 0
    best_f1_val = 0
    for i, row in enumerate(rows):
        try:
            f1_val = float(row[3])
            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_f1_idx = i
        except (ValueError, IndexError):
            pass

    for j in range(len(headers)):
        table[best_f1_idx + 1, j].set_facecolor("#C8E6C9")

    ax.set_title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved summary table image to results/comparison/summary_table.png")


if __name__ == "__main__":
    print("=" * 50)
    print("  Model Comparison Report Generator")
    print("=" * 50)
    print()

    os.makedirs(COMPARISON_DIR, exist_ok=True)

    print("loading results...")
    models = load_all_results()

    if len(models) == 0:
        print("\nno results found. train some models first:")
        print("  1. set MODEL_TYPE in config.py, run bash run.sh")
        print("  2. copy results/evaluation_results.json to results/<model>_results.json")
        print("  3. repeat for each model type")
        print("  4. run python train_yolo.py for YOLO comparison")
        print("  5. run this script again")
        exit()

    print(f"\nfound results for {len(models)} models: {', '.join(models.keys())}")

    # generate all comparison outputs
    print("\ngenerating comparison charts...")
    plot_model_comparison(models)
    plot_improvement_journey(models)
    generate_summary_table(models)

    print(f"\ndone! check results/comparison/ for all outputs")
