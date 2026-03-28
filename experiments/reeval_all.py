# experiments/reeval_all.py
# re-evaluates all saved checkpoints without retraining
# useful when evaluation code was fixed but models dont need to retrain
#
# usage: python experiments/reeval_all.py

import sys
import os
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

os.environ["PYTHONUNBUFFERED"] = "1"

SAVED_RESULTS = os.path.join(PROJECT_ROOT, "saved_results")

# models to re-evaluate and their config overrides
MODELS = {
    "custom_ssd": {"MODEL_TYPE": "custom_ssd", "BATCH_SIZE": 16},
    "pretrained_ssd": {"MODEL_TYPE": "pretrained_ssd", "BATCH_SIZE": 24},
    "retinanet": {"MODEL_TYPE": "retinanet", "BATCH_SIZE": 8, "RETINANET_INPUT_SIZE": 512},
}

# ablation variants
ABLATION_VARIANTS = {
    "full_pipeline": {"MODEL_TYPE": "retinanet", "BATCH_SIZE": 8, "RETINANET_INPUT_SIZE": 512},
    "hsv_only": {"MODEL_TYPE": "retinanet", "BATCH_SIZE": 8, "RETINANET_INPUT_SIZE": 512},
    "rgb_only": {"MODEL_TYPE": "retinanet", "BATCH_SIZE": 8, "RETINANET_INPUT_SIZE": 512},
}


def reeval_model(name, config_overrides, checkpoint_dir, plots_dir, output_dir):
    # re-evaluates a single model from its saved checkpoint

    # check if checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  {name}: no checkpoint found at {checkpoint_path}, skipping")
        return None

    print(f"\n{'=' * 40}")
    print(f"  Re-evaluating: {name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'=' * 40}", flush=True)

    # apply config overrides
    import config
    for key, value in config_overrides.items():
        setattr(config, key, value)

    # run evaluation
    from evaluate import run_evaluation
    metrics = run_evaluation(output_dir=plots_dir, checkpoint_dir=checkpoint_dir)

    # save metrics.json
    if metrics:
        eval_results = os.path.join(plots_dir, "evaluation_results.json")
        if os.path.exists(eval_results):
            shutil.copy(eval_results, os.path.join(output_dir, "metrics.json"))
        print(f"  {name}: P={metrics.get('precision',0):.4f} R={metrics.get('recall',0):.4f} F1={metrics.get('f1',0):.4f}")

    return metrics


if __name__ == "__main__":
    print("=" * 50)
    print("  Re-evaluating all saved checkpoints")
    print("=" * 50)

    # re-evaluate each model
    for name, overrides in MODELS.items():
        model_dir = os.path.join(SAVED_RESULTS, name)
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        plots_dir = os.path.join(model_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        reeval_model(name, overrides, checkpoint_dir, plots_dir, model_dir)

    # re-evaluate ablation variants
    for variant_name, overrides in ABLATION_VARIANTS.items():
        variant_dir = os.path.join(SAVED_RESULTS, "ablation", variant_name)
        checkpoint_dir = os.path.join(variant_dir, "checkpoints")
        plots_dir = os.path.join(variant_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        reeval_model(f"ablation/{variant_name}", overrides, checkpoint_dir, plots_dir, variant_dir)

    # regenerate comparison report
    print("\n--- Generating Report ---", flush=True)
    from generate_report import find_all_results, find_ablation_results
    from generate_report import plot_model_comparison, plot_improvement_journey
    from generate_report import generate_summary_table, save_combined_results

    REPORT_DIR = os.path.join(SAVED_RESULTS, "report")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # need to set the module level REPORT_DIR
    import generate_report
    generate_report.REPORT_DIR = REPORT_DIR

    models = find_all_results()
    ablation = find_ablation_results()
    if len(models) >= 2:
        plot_model_comparison(models)
        plot_improvement_journey(models)
        generate_summary_table(models)
        save_combined_results(models, ablation)

    print(f"\n{'=' * 50}")
    print(f"  Done! Check saved_results/ for updated results")
    print(f"{'=' * 50}")
