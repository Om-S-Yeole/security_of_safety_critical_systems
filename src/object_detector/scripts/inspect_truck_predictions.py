from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

# ===============================================================================
# Paths
# ===============================================================================

MODEL_PATH = Path("./runs/detect/runs/yolov8m_baseline-2/weights/best.pt")

VAL_IMAGES_DIR = Path("./project_data/validation/images/")
VAL_LABELS_DIR = Path("./project_data/validation/labels/")

OUTPUT_DIR = Path("./runs/detect/truck_diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Truck = class 2 according to dataset.yaml
TRUCK_CLASS_ID = 2

# Detection confidence threshold
CONF_THRESHOLD = 0.01


# ===============================================================================
# Helper: read YOLO ground-truth labels
# ===============================================================================


def load_labels(label_path, image_width, image_height):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, x_center, y_center, width, height = map(float, values)

            class_id = int(class_id)

            # We only care about Trucks
            if class_id != TRUCK_CLASS_ID:
                continue

            # Convert normalized YOLO coordinates to pixel coordinates
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            boxes.append((x1, y1, x2, y2))

    return boxes


# ===============================================================================
# Main
# ===============================================================================


def main():

    model = YOLO(MODEL_PATH)

    image_paths = sorted(
        list(VAL_IMAGES_DIR.glob("*.jpg"))
        + list(VAL_IMAGES_DIR.glob("*.jpeg"))
        + list(VAL_IMAGES_DIR.glob("*.png"))
    )

    print(f"Found {len(image_paths)} validation images.\n")

    truck_image_count = 0

    for image_path in image_paths:
        # -----------------------------------------------------------------------
        # Load image
        # -----------------------------------------------------------------------

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # -----------------------------------------------------------------------
        # Load ground-truth Truck boxes
        # -----------------------------------------------------------------------

        label_path = VAL_LABELS_DIR / f"{image_path.stem}.txt"

        gt_trucks = load_labels(
            label_path,
            width,
            height,
        )

        # Ignore images without Trucks
        if len(gt_trucks) == 0:
            continue

        truck_image_count += 1

        print("=" * 70)
        print(f"Image: {image_path.name}")
        print(f"Ground-truth Trucks: {len(gt_trucks)}")

        # -----------------------------------------------------------------------
        # Run YOLO
        # -----------------------------------------------------------------------

        results = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        # -----------------------------------------------------------------------
        # Print predictions
        # -----------------------------------------------------------------------

        print("\nPredictions:")

        if result.boxes is None or len(result.boxes) == 0:
            print("  No predictions.")
        else:
            for i in range(len(result.boxes)):
                cls = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())

                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

                class_name = model.names[cls]

                print(
                    f"  {i + 1}: "
                    f"{class_name:<10} "
                    f"confidence={conf:.3f} "
                    f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                )

        # -----------------------------------------------------------------------
        # Draw image
        # -----------------------------------------------------------------------

        draw = ImageDraw.Draw(image)

        # Ground truth = green
        for box in gt_trucks:
            x1, y1, x2, y2 = box

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="green",
                width=4,
            )

            draw.text(
                (x1, max(0, y1 - 20)),
                "GT Truck",
                fill="green",
            )

        # Predictions
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                cls = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())

                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                class_name = model.names[cls]

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3,
                )

                draw.text(
                    (x1, y1),
                    f"{class_name} {conf:.2f}",
                    fill="red",
                )

        # -----------------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------------

        output_path = OUTPUT_DIR / image_path.name
        image.save(output_path)

        print(f"\nSaved: {output_path}")

    print("\n" + "=" * 70)
    print(f"Images containing Trucks: {truck_image_count}")
    print(f"Diagnostic images saved to: {OUTPUT_DIR}")


# ===============================================================================
# Run
# ===============================================================================

if __name__ == "__main__":
    main()
