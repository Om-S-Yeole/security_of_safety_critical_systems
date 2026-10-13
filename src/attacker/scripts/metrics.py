from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def _to_float(value) -> float | None:
    """Convert a scalar-like value to float, or return None."""
    if value is None:
        return None

    # Supports Python scalars and scalar tensors.
    if hasattr(value, "item"):
        value = value.item()

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_yolo_metrics(
    validation_results,
    class_names: Mapping[int, str] | None = None,
) -> dict[str,]:
    """
    Extract standard detection metrics from Ultralytics validation output.

    Parameters
    ----------
    validation_results:
        The metrics object returned by an Ultralytics YOLO validation
        run, for example:
            results = model.val(data="data.yaml")

    class_names:
        Optional mapping from class ID to class name, e.g.:
            {0: "Aircraft", 1: "Building", 2: "Truck", 3: "Tank"}

    Returns
    -------
    dict
        A JSON-serializable dictionary of summary and per-class metrics.
    """
    box_metrics = getattr(validation_results, "box", None)

    if box_metrics is None:
        raise ValueError(
            "No bounding-box metrics found. Pass the metrics object "
            "returned by Ultralytics model.val()."
        )

    summary = {
        "precision": _to_float(getattr(box_metrics, "mp", None)),
        "recall": _to_float(getattr(box_metrics, "mr", None)),
        "map50": _to_float(getattr(box_metrics, "map50", None)),
        "map50_95": _to_float(getattr(box_metrics, "map", None)),
    }

    # Include AP per class when available.
    per_class: dict[str, dict[str,]] = {}

    ap_values = getattr(box_metrics, "ap", None)
    class_indices = getattr(box_metrics, "ap_class_index", None)

    if ap_values is not None and class_indices is not None:
        if hasattr(ap_values, "tolist"):
            ap_values = ap_values.tolist()

        if hasattr(class_indices, "tolist"):
            class_indices = class_indices.tolist()

        for class_id, ap_row in zip(class_indices, ap_values):
            class_id = int(class_id)

            if hasattr(ap_row, "tolist"):
                ap_row = ap_row.tolist()

            if not isinstance(ap_row, (list, tuple)):
                ap_row = [ap_row]

            # Ultralytics AP columns normally correspond to IoU thresholds
            # from 0.50 through 0.95 in increments of 0.05.
            ap50 = _to_float(ap_row[0]) if len(ap_row) >= 1 else None
            ap50_95 = (
                sum(float(v) for v in ap_row) / len(ap_row) if len(ap_row) > 0 else None
            )

            name = (
                class_names.get(class_id, str(class_id))
                if class_names is not None
                else str(class_id)
            )

            per_class[name] = {
                "class_id": class_id,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }

    return {
        "metrics": summary,
        "per_class": per_class,
    }


def save_metrics(
    metrics: dict[str,],
    output_path: str | Path,
) -> Path:
    """
    Save a metrics dictionary as formatted JSON.

    Parameters
    ----------
    metrics:
        Dictionary returned by summarize_yolo_metrics().
    output_path:
        Destination JSON file path.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)

    return output_path


def print_metrics(metrics: dict[str,]) -> None:
    """Print the summary metrics in a readable format."""
    summary = metrics.get("metrics", {})

    print("\nObject Detection Metrics")
    print("-" * 32)

    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP@0.50",
        "map50_95": "mAP@0.50:0.95",
    }

    for key, label in labels.items():
        value = summary.get(key)
        if value is None:
            print(f"{label:<18}: N/A")
        else:
            print(f"{label:<18}: {value:.4f}")

    per_class = metrics.get("per_class", {})

    if per_class:
        print("\nPer-Class Average Precision")
        print("-" * 48)

        for name, values in per_class.items():
            ap50 = values.get("ap50")
            ap50_95 = values.get("ap50_95")

            ap50_text = f"{ap50:.4f}" if ap50 is not None else "N/A"
            ap_text = f"{ap50_95:.4f}" if ap50_95 is not None else "N/A"

            print(f"{name:<20} AP@0.50={ap50_text:<8} AP@0.50:0.95={ap_text}")


if __name__ == "__main__":
    print(
        "metrics.py provides helper functions for Ultralytics validation "
        "results. Import summarize_yolo_metrics(), save_metrics(), and "
        "print_metrics() into your evaluation script."
    )
