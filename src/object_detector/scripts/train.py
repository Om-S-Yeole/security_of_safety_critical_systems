from pathlib import Path

from ultralytics import YOLO

from src.helpers import load_yaml

# ===============================================================================
# Load the YAML
# ===============================================================================
TRAINING_YAML_PATH = Path("./src/object_detector/configs/train_balanced.yaml")
training_configs = load_yaml(TRAINING_YAML_PATH)

# ===============================================================================
# Model training
# ===============================================================================

def main():
    # Load pretrained YOLOv8m
    model = YOLO("yolov8m.pt")

    # Fine-tune on our dataset
    model.train(**training_configs["train_configs"])

# ===============================================================================
# Start training
# ===============================================================================

if __name__ == "__main__":
    main()