from pathlib import Path

from ultralytics import YOLO

from src.attacker.scripts.metrics import (
    print_metrics,
    save_metrics,
    summarize_yolo_metrics,
)

model = YOLO("detector/weights/best.pt")

results = model.val(
    data="src/attacker/configs/dataset_balanced.yaml",
    split="test",
)

class_names = {
    0: "Aircraft",
    1: "Building",
    2: "Truck",
    3: "Tank",
}

metrics = summarize_yolo_metrics(results, class_names)
print_metrics(metrics)

save_metrics(
    metrics,
    Path("experiments/run_001/baseline_metrics.json"),
)
