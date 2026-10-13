from pathlib import Path

from ultralytics import YOLO

from src.helpers import load_yaml

TESTING_YAML_PATH = Path("./src/object_detector/configs/test.yaml")
test_yaml = load_yaml(TESTING_YAML_PATH)


def main():
    model = YOLO(test_yaml["model_weights_path"])

    metrics = model.val(
        data=test_yaml["test_configs"]["data"],
        split="test",
        imgsz=test_yaml["test_configs"]["imgsz"],
        device=test_yaml["test_configs"]["device"],
    )

    print("\nTest Evaluation Complete")
    print(f"Precision:      {metrics.box.mp:.3f}")
    print(f"Recall:         {metrics.box.mr:.3f}")
    print(f"mAP@50:         {metrics.box.map50:.3f}")
    print(f"mAP@50-95:      {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
