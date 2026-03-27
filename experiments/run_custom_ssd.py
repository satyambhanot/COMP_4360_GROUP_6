# experiments/run_custom_ssd.py
# trains and evaluates our custom SSD with MobileNetV2 backbone
# saves everything to saved_results/custom_ssd/
#
# usage: python experiments/run_custom_ssd.py

from base_experiment import Experiment

if __name__ == "__main__":
    exp = Experiment(
        name="custom_ssd",
        config_overrides={
            "MODEL_TYPE": "custom_ssd",
            "BATCH_SIZE": 16,
            "NUM_EPOCHS": 150,
            "EARLY_STOPPING_PATIENCE": 25,
        }
    )
    exp.run()
