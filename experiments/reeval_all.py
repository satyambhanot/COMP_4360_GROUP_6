'''
reeval_all.py
re-evaluates all saved checkpoints without retraining
useful when evaluation code was fixed but models dont need to retrain
ablation variants spawn as separate processes for correct preprocessing flags

usage: python experiments/reeval_all.py
'''

import sys
import os
import json
import shutil
import subprocess
import time

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


def reeval_model(name, config_overrides, checkpoint_dir, plots_dir, output_dir):
    # re-evaluates a single model from its saved checkpoint

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  {name}: no checkpoint found, skipping")
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


def reeval_ablation_variant(variant_name, use_hsv, use_sobel):
    # re-evaluates one ablation variant as a separate process
    # needs its own process because the model has different input channels
    # depending on use_hsv and use_sobel flags

    script = os.path.join(PROJECT_ROOT, "experiments", "reeval_ablation_variant.py")
    cmd = [sys.executable, script, variant_name, str(use_hsv), str(use_sobel)]

    print(f"\n  launching: {variant_name} (hsv={use_hsv}, sobel={use_sobel})", flush=True)
    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})

    return result.returncode == 0


if __name__ == "__main__":
    print("=" * 50)
    print("  Re-evaluating all saved checkpoints")
    print("=" * 50)

    # re-evaluate each main model
    for name, overrides in MODELS.items():
        model_dir = os.path.join(SAVED_RESULTS, name)
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        plots_dir = os.path.join(model_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        reeval_model(name, overrides, checkpoint_dir, plots_dir, model_dir)

    # re-evaluate ablation variants as separate processes
    # each variant has different preprocessing flags which change the model architecture
    ablation_variants = [
        ("full_pipeline", True, True),
        ("hsv_only", True, False),
        ("rgb_only", False, False),
    ]

    for variant_name, use_hsv, use_sobel in ablation_variants:
        variant_dir = os.path.join(SAVED_RESULTS, "ablation", variant_name)
        if os.path.exists(os.path.join(variant_dir, "checkpoints", "best_model.pth")):
            reeval_ablation_variant(variant_name, use_hsv, use_sobel)
            time.sleep(3)
        else:
            print(f"  {variant_name}: no checkpoint, skipping")

    # regenerate comparison report
    print("\n--- Generating Report ---", flush=True)
    os.system(f"{sys.executable} {os.path.join(PROJECT_ROOT, 'experiments', 'generate_report.py')}")

    print(f"\n{'=' * 50}")
    print(f"  Done! Check saved_results/ for updated results")
    print(f"{'=' * 50}")
