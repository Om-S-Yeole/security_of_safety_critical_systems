from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# Configuration
# ============================================================


DATASET_DIR = Path("./project_data")

SPLITS = {
    "train": DATASET_DIR / "train",
    "validation": DATASET_DIR / "validation",
    "test": DATASET_DIR / "test",
}

CLASS_NAMES = {
    0: "Aircraft",
    1: "Building",
    2: "Truck",
    3: "Tank",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# Helper functions
# ============================================================


def find_images(image_dir):
    """Return all image files in a directory."""
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(label_dir):
    """Return all YOLO label files in a directory."""
    return sorted(label_dir.glob("*.txt"))


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Analyze one split
# ============================================================


def analyze_split(split_name, split_dir):
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    # Your current dataset uses lowercase paths according to
    # the validation output. If the directories don't exist,
    # try the original capitalized naming.
    if not image_dir.exists():
        image_dir = split_dir / "Images"

    if not label_dir.exists():
        label_dir = split_dir / "Labels"

    print_header(f"{split_name.upper()} DATASET")

    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        return

    if not label_dir.exists():
        print(f"ERROR: Label directory not found: {label_dir}")
        return

    images = find_images(image_dir)
    labels = find_labels(label_dir)

    print(f"Image directory : {image_dir}")
    print(f"Label directory : {label_dir}")
    print(f"Images          : {len(images)}")
    print(f"Label files     : {len(labels)}")

    # --------------------------------------------------------
    # Image-label correspondence
    # --------------------------------------------------------

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)

    print()
    print("IMAGE/LABEL CORRESPONDENCE")
    print("-" * 40)
    print(f"Images without labels : {len(images_without_labels)}")
    print(f"Labels without images : {len(labels_without_images)}")

    if images_without_labels:
        print("\nImages without labels:")
        for name in images_without_labels:
            print(f"  {name}")

    if labels_without_images:
        print("\nLabels without images:")
        for name in labels_without_images:
            print(f"  {name}")

    # --------------------------------------------------------
    # Label analysis
    # --------------------------------------------------------

    class_counts = Counter()

    # Store normalized bounding-box dimensions.
    widths = defaultdict(list)
    heights = defaultdict(list)
    areas = defaultdict(list)

    invalid_label_files = []
    invalid_rows = []

    total_objects = 0

    for label_path in labels:
        try:
            with label_path.open("r") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            invalid_label_files.append((label_path.name, str(exc)))
            continue

        for line_number, line in enumerate(lines, start=1):
            values = line.split()

            # A YOLO detection row MUST contain:
            #
            # class x_center y_center width height
            #
            # = exactly 5 values
            if len(values) != 5:
                invalid_rows.append(
                    (
                        label_path.name,
                        line_number,
                        f"Expected 5 values, found {len(values)}",
                        line,
                    )
                )
                continue

            try:
                class_id = int(values[0])
                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])
            except ValueError:
                invalid_rows.append(
                    (
                        label_path.name,
                        line_number,
                        "Non-numeric value",
                        line,
                    )
                )
                continue

            # ------------------------------------------------
            # Validate class ID
            # ------------------------------------------------

            if class_id not in CLASS_NAMES:
                invalid_rows.append(
                    (
                        label_path.name,
                        line_number,
                        f"Invalid class ID: {class_id}",
                        line,
                    )
                )
                continue

            # ------------------------------------------------
            # Validate normalized coordinates
            # ------------------------------------------------

            if not (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                invalid_rows.append(
                    (
                        label_path.name,
                        line_number,
                        "Bounding-box values outside valid range",
                        line,
                    )
                )
                continue

            # ------------------------------------------------
            # Valid object
            # ------------------------------------------------

            class_counts[class_id] += 1
            total_objects += 1

            widths[class_id].append(width)
            heights[class_id].append(height)

            # Normalized bounding-box area.
            areas[class_id].append(width * height)

    # --------------------------------------------------------
    # Class statistics
    # --------------------------------------------------------

    print()
    print("CLASS DISTRIBUTION")
    print("-" * 40)

    print(f"{'Class':<18}{'ID':<6}{'Objects':>10}")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name:<18}{class_id:<6}{class_counts[class_id]:>10}")

    print(f"\nTotal objects: {total_objects}")

    # --------------------------------------------------------
    # Bounding-box statistics
    # --------------------------------------------------------

    print()
    print("BOUNDING-BOX SIZE STATISTICS")
    print("-" * 70)

    print(
        f"{'Class':<18}{'Mean W':>10}{'Mean H':>10}{'Mean Area':>12}{'Median Area':>14}"
    )

    for class_id, class_name in CLASS_NAMES.items():
        if not areas[class_id]:
            print(f"{class_name:<18} No valid objects")
            continue

        mean_width = np.mean(widths[class_id])
        mean_height = np.mean(heights[class_id])
        mean_area = np.mean(areas[class_id])
        median_area = np.median(areas[class_id])

        print(
            f"{class_name:<18}"
            f"{mean_width:>10.4f}"
            f"{mean_height:>10.4f}"
            f"{mean_area:>12.5f}"
            f"{median_area:>14.5f}"
        )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    print()
    print("IMAGE DIMENSIONS")
    print("-" * 50)

    dimension_counts = Counter()
    unreadable_images = []

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            unreadable_images.append(image_path.name)
            continue

        height, width = image.shape[:2]
        dimension_counts[(width, height)] += 1

    for (width, height), count in dimension_counts.most_common():
        print(f"{width} x {height}: {count} images")

    if unreadable_images:
        print("\nUnreadable images:")
        for name in unreadable_images:
            print(f"  {name}")

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    print()
    print("LABEL VALIDATION")
    print("-" * 40)

    print(f"Invalid label files : {len(invalid_label_files)}")
    print(f"Invalid label rows  : {len(invalid_rows)}")

    if invalid_label_files:
        print("\nInvalid label files:")
        for filename, error in invalid_label_files:
            print(f"  {filename}: {error}")

    if invalid_rows:
        print("\nInvalid label rows:")
        for filename, line_number, reason, line in invalid_rows:
            print(f"  {filename}:{line_number} → {reason} → {line}")


# ============================================================
# Main
# ============================================================


def main():

    print_header("YOLO DATASET ANALYSIS")

    print("\nClass mapping:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}")

    for split_name, split_dir in SPLITS.items():
        analyze_split(split_name, split_dir)

    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
