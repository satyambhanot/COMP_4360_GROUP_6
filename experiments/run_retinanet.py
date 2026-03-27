# experiments/run_retinanet.py
# trains and evaluates the RetinaNet model
# saves everything to saved_results/retinanet/
#
# usage: python experiments/run_retinanet.py

from base_experiment import Experiment

if __name__ == "__main__":
    exp = Experiment(
        name="RetinaNet",
        config_overrides={
            "MODEL_TYPE": "retinanet",
            "BATCH_SIZE": 8,
            "RETINANET_INPUT_SIZE": 512,
            "NUM_EPOCHS": 150,
            "EARLY_STOPPING_PATIENCE": 25,
        }
    )
    exp.run()
