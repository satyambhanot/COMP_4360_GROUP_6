'''
run_pretrained_ssd.py
trains and evaluates the pretrained SSD300 VGG16 model
saves everything to saved_results/pretrained_ssd/

usage: python experiments/run_pretrained_ssd.py
'''

from base_experiment import Experiment

if __name__ == "__main__":
    exp = Experiment(
        name="pretrained_ssd",
        config_overrides={
            "MODEL_TYPE": "pretrained_ssd",
            "BATCH_SIZE": 24,
            "NUM_EPOCHS": 150,
            "EARLY_STOPPING_PATIENCE": 25,
        }
    )
    exp.run()
