'''
run_ablation_color_variant.py
trains and evaluates a single color information variant
called by run_ablation_color.py as a separate process for clean GPU memory

for grayscale and binary variants, a wrapper converts images before the model
sees them. the wrapper is applied during both training and evaluation so the
model always sees the same type of input.

usage: python experiments/run_ablation_color_variant.py <name> <color_mode>
  color_mode: "rgb", "gray", or "binary"
'''

import sys
import os
import json

os.environ["PYTHONUNBUFFERED"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

'''
override config for RetinaNet with reduced epochs for ablation
must happen before importing train/evaluate so they pick up the changes
'''
import config
config.MODEL_TYPE = "retinanet"
config.BATCH_SIZE = 8
config.NUM_EPOCHS = 30
config.EARLY_STOPPING_PATIENCE = 15

import torch
import torch.nn as nn
from config import IMAGENET_MEAN, IMAGENET_STD

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "saved_results", "ablation_color")


class ColorTransformWrapper(nn.Module):
    '''
    wraps a detection model to convert images before the model sees them
    supports three modes:
      rgb: no conversion (passthrough)
      gray: convert to grayscale, copy to 3 channels
      binary: convert to grayscale, threshold at 0.5, copy to 3 channels

    the conversion undoes imagenet normalization first, applies the color
    transform, then re-normalizes so the backbone gets valid input
    '''

    def __init__(self, base_model, color_mode):
        super().__init__()
        self.base_model = base_model
        self.color_mode = color_mode

        '''
        copy attributes from the base model so train.py and evaluate.py
        can identify the model type and architecture correctly
        '''
        self.model_type = base_model.model_type
        self.use_hsv = getattr(base_model, "use_hsv", False)
        self.use_sobel = getattr(base_model, "use_sobel", False)

        '''
        register imagenet normalization constants as buffers
        so they move to the right device automatically with the model
        '''
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _convert_color(self, images):
        '''converts imagenet-normalized images to the target color mode'''

        if self.color_mode == "rgb":
            return images

        # undo imagenet normalization to get 0-1 range RGB
        rgb = (images * self.img_std + self.img_mean).clamp(0, 1)

        # standard luminance weights for RGB to grayscale
        gray = 0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]

        if self.color_mode == "binary":
            gray = (gray > 0.5).float()

        # copy single channel to 3 channels for backbone compatibility
        out = gray.repeat(1, 3, 1, 1)

        # re-apply imagenet normalization
        out = (out - self.img_mean) / self.img_std
        return out

    def forward(self, images, targets=None):
        '''converts images then passes through to the base model'''
        images = self._convert_color(images)
        return self.base_model(images, targets)

    def get_param_groups(self):
        '''delegates to the base model for optimizer parameter groups'''
        return self.base_model.get_param_groups()

    def state_dict(self, *args, **kwargs):
        '''returns the base model state dict so checkpoints are clean'''
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        '''loads into the base model'''
        return self.base_model.load_state_dict(*args, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python run_ablation_color_variant.py <name> <color_mode>")
        print("example: python run_ablation_color_variant.py color_gray gray")
        exit(1)

    variant_name = sys.argv[1]
    color_mode = sys.argv[2]

    variant_dir = os.path.join(OUTPUT_BASE, variant_name)
    checkpoint_dir = os.path.join(variant_dir, "checkpoints")
    plots_dir = os.path.join(variant_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"{'=' * 50}")
    print(f"  Ablation Variant: {variant_name}")
    print(f"  Color Mode: {color_mode}")
    print(f"  Output: {variant_dir}")
    print(f"{'=' * 50}", flush=True)

    '''
    for rgb mode: use full preprocessing (HSV + Sobel) as the control
    for gray/binary: disable HSV preprocessing since color info is removed
    and the HSV mask would be meaningless on grayscale images
    '''
    from models.model import create_model

    if color_mode == "rgb":
        model = create_model(use_hsv=True, use_sobel=True)
    else:
        model = create_model(use_hsv=False, use_sobel=False)

    # wrap the model with color conversion
    model = ColorTransformWrapper(model, color_mode)

    # train
    print("\n--- Training ---", flush=True)
    from train import run_training
    model, val_loss, history = run_training(
        checkpoint_dir=checkpoint_dir,
        model_override=model
    )

    # free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    '''
    for evaluation we need to reload the checkpoint and re-wrap it
    because run_evaluation loads the model fresh from the checkpoint file
    for rgb mode the standard load works fine since the base model is normal
    for gray/binary we use model_override to pass in the wrapped model
    '''
    print("\n--- Evaluation ---", flush=True)
    from evaluate import run_evaluation, load_model

    if color_mode == "rgb":
        metrics = run_evaluation(output_dir=plots_dir, checkpoint_dir=checkpoint_dir)
    else:
        '''
        temporarily point config at our checkpoint dir so load_model finds it
        then wrap the loaded model with the same color conversion used in training
        '''
        original_ckpt_dir = config.CHECKPOINTS_DIR
        config.CHECKPOINTS_DIR = checkpoint_dir
        base_model = load_model()
        config.CHECKPOINTS_DIR = original_ckpt_dir

        if base_model is not None:
            wrapped_model = ColorTransformWrapper(base_model, color_mode)
            wrapped_model = wrapped_model.to(next(base_model.parameters()).device)
            wrapped_model.eval()
            metrics = run_evaluation(
                output_dir=plots_dir,
                checkpoint_dir=checkpoint_dir,
                model_override=wrapped_model
            )
        else:
            metrics = None

    # save metrics
    if metrics:
        result = {
            "variant": variant_name,
            "color_mode": color_mode,
            "metrics": metrics
        }
        with open(os.path.join(variant_dir, "metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n  {variant_name} complete!", flush=True)
