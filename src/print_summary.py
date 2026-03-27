# print_summary.py
# prints model summary and training results
# called by run.sh to keep the bash script clean
# these were originally inline Python blocks in run.sh

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from model import create_model
from config import MODEL_TYPE, CHECKPOINTS_DIR, NUM_EPOCHS


def print_model_summary():
    # prints the model architecture, preprocessing, and parameter counts
    model = create_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"model type: {model.model_type}")

    if MODEL_TYPE == "retinanet":
        print(f"  architecture: RetinaNet ResNet50 FPN v2 (pretrained on COCO)")
        print(f"  preprocessing: HSV mask + Sobel edge detection on HSV")
        print(f"  backbone: ResNet50 + FPN (frozen, pretrained on COCO)")
        print(f"  detection heads: RetinaNet heads with built in focal loss")
        print(f"  input size: 512x512 (larger view for more context)")
    elif MODEL_TYPE == "pretrained_ssd":
        print(f"  architecture: SSD300 VGG16 (pretrained on COCO)")
        print(f"  preprocessing: HSV mask + Sobel edge detection on HSV")
        print(f"  backbone: VGG16 (frozen, pretrained on COCO)")
        print(f"  detection heads: SSD heads (classification head replaced for waldo)")
        print(f"  input size: 300x300")
    else:
        print(f"  architecture: Custom WaldoDetector")
        print(f"  preprocessing: HSV mask + Sobel edge detection on HSV")
        print(f"  backbone: MobileNetV2 (first 11 layers, frozen)")
        print(f"  detection heads: 5 custom SSD blocks")
        print(f"  input size: 256x256")

    print()
    print(f"parameters:")
    print(f"  total:     {total:,}")
    print(f"  trainable: {trainable:,}")
    print(f"  frozen:    {frozen:,}")


def print_run_summary():
    # prints the final results after evaluation is done
    # loads the evaluation_results.json and formats it nicely
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "evaluation_results.json")

    if not os.path.exists(results_path):
        print("no evaluation results found")
        return

    with open(results_path) as f:
        results = json.load(f)

    m = results["metrics"]
    model_names = {
        "retinanet": "RetinaNet ResNet50 FPN",
        "pretrained_ssd": "pretrained SSD300",
        "custom_ssd": "custom WaldoDetector"
    }
    print(f"model type: {model_names.get(MODEL_TYPE, MODEL_TYPE)}")
    print(f"epochs trained: {NUM_EPOCHS}")
    print()
    print(f"test set results:")
    print(f"  accuracy:  {m['accuracy']:.4f}")
    print(f"  precision: {m['precision']:.4f}")
    print(f"  recall:    {m['recall']:.4f}")
    print(f"  f1 score:  {m['f1']:.4f}")
    print(f"  true positives:  {m['tp']}")
    print(f"  false positives: {m['fp']}")
    print(f"  true negatives:  {m['tn']}")
    print(f"  false negatives: {m['fn']}")
    print()
    print(f"confidence stats:")
    print(f"  avg positive tile: {results['avg_confidence_positive']:.4f}")
    print(f"  avg negative tile: {results['avg_confidence_negative']:.4f}")
    print(f"  max positive tile: {results['max_confidence_positive']:.4f}")
    print(f"  max negative tile: {results['max_confidence_negative']:.4f}")


if __name__ == "__main__":
    # when called with an argument, run that specific function
    if len(sys.argv) > 1:
        if sys.argv[1] == "model":
            print_model_summary()
        elif sys.argv[1] == "results":
            print_run_summary()
    else:
        print_model_summary()
