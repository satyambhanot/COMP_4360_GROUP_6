# experiments/base_experiment.py
# base class for all experiment scripts
# handles the common boilerplate: config overrides, directory setup,
# train -> evaluate -> save flow
#
# each experiment just needs to specify its name and config overrides
# the base class does everything else

import sys
import os
import shutil

# force python to flush output immediately so progress shows in real time
os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


class Experiment:
    # base class that handles the full train -> evaluate -> save pipeline
    # subclasses just set the name and config, this class does the rest

    def __init__(self, name, config_overrides):
        self.name = name
        self.config_overrides = config_overrides

        # set up output directories inside saved_results/
        self.output_dir = os.path.join(PROJECT_ROOT, "saved_results", name.lower().replace(" ", "_"))
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.plots_dir = os.path.join(self.output_dir, "plots")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def apply_config(self):
        # override config values before importing anything that reads them
        # this is called before train.py or evaluate.py load their config
        import config
        for key, value in self.config_overrides.items():
            setattr(config, key, value)

    def run(self):
        # the full experiment pipeline: apply config -> train -> evaluate -> save
        print(f"{'=' * 50}")
        print(f"  Experiment: {self.name}")
        print(f"  Output: {self.output_dir}")
        print(f"{'=' * 50}")
        print(flush=True)

        # apply config overrides
        self.apply_config()

        # train the model
        print("--- Training ---", flush=True)
        from train import run_training
        model, best_val_loss, history = run_training(checkpoint_dir=self.checkpoint_dir)

        # free GPU memory before evaluation
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # evaluate on test set
        print("\n--- Evaluation ---", flush=True)
        from evaluate import run_evaluation
        metrics = run_evaluation(output_dir=self.plots_dir, checkpoint_dir=self.checkpoint_dir)

        # copy the main metrics json to the top level of the output dir
        # so generate_report.py can find it easily
        eval_results = os.path.join(self.plots_dir, "evaluation_results.json")
        if os.path.exists(eval_results):
            shutil.copy(eval_results, os.path.join(self.output_dir, "metrics.json"))

        print(f"\n{'=' * 50}")
        print(f"  {self.name} experiment complete!")
        print(f"  Results saved to: {self.output_dir}")
        print(f"{'=' * 50}")

        return metrics
