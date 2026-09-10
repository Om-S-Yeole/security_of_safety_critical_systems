from pathlib import Path

from ultralytics import YOLO

from src.helpers import load_yaml

# ===============================================================================
# Define the Paths
# ===============================================================================
VALIDATION_YAML_PATH = Path("./src/object_detector/configs/validate.yaml")

# ===============================================================================
# Load the YAML
# ===============================================================================
val_yaml = load_yaml(VALIDATION_YAML_PATH)

# ===============================================================================
# Define main function
# ===============================================================================


def main():
    model = YOLO(val_yaml["model_weights_path"])

    metrics = model.val(**val_yaml["validate_configs"])

    print("Validation Complete\n")


# ===============================================================================
# Run the main function
# ===============================================================================
if __name__ == "__main__":
    main()
