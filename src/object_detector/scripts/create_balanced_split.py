import random
import shutil
from collections import Counter
from filecmp import cmp
from pathlib import Path

# ===============================================================================
# Configuration
# ===============================================================================

DATASET_DIR = Path("./project_data")
OUTPUT_DIR = DATASET_DIR / "balanced_split"

SOURCE_SPLITS = ("train", "validation", "test")
OUTPUT_SPLITS = ("train", "validation", "test")

# Target proportions for the output splits.
SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}

# Reproducible random selection.
RANDOM_SEED = 42

# Dataset classes.
CLASS_NAMES = {
    0: "Aircraft",
    1: "Building",
    2: "Truck",
    3: "Tank",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ===============================================================================
# Helper functions
# ===============================================================================


def get_image_files(split):
    """Return all images in a source split."""

    images_dir = DATASET_DIR / split / "images"

    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_classes_in_image(label_path):
    """Return the set of classes present in an image."""

    classes = set()

    if not label_path.exists():
        return classes

    with label_path.open() as label_file:
        for line in label_file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            classes.add(int(values[0]))

    return classes


def collect_image_files():
    """Collect images and their labels from every original dataset split."""

    image_files = []

    for split in SOURCE_SPLITS:
        for image_path in get_image_files(split):
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            image_files.append((split, image_path, label_path))

    unique_images = {}

    for image_record in image_files:
        _, image_path, label_path = image_record
        existing_record = unique_images.get(image_path.name)

        if existing_record is None:
            unique_images[image_path.name] = image_record
            continue

        _, existing_image_path, existing_label_path = existing_record
        same_image = cmp(image_path, existing_image_path, shallow=False)
        same_label = (
            label_path.exists()
            and existing_label_path.exists()
            and cmp(label_path, existing_label_path, shallow=False)
        )

        if not (same_image and same_label):
            raise RuntimeError(
                f"Different files use the same image filename: {image_path.name}"
            )

    return list(unique_images.values())


def get_split_counts(total_images):
    """Return integer split sizes that are as close as possible to the ratios."""

    split_counts = {
        split: int(total_images * ratio)
        for split, ratio in SPLIT_RATIOS.items()
    }
    remainder = total_images - sum(split_counts.values())

    # Assign rounding leftovers to the largest split so the requested
    # proportions remain closest to the target.
    for split in sorted(
        SPLIT_RATIOS,
        key=lambda name: SPLIT_RATIOS[name],
        reverse=True,
    ):
        if remainder == 0:
            break
        split_counts[split] += 1
        remainder -= 1

    return split_counts


def create_balanced_splits(image_files):
    """Assign every pooled image to a split while balancing class presence."""

    split_counts = get_split_counts(len(image_files))
    class_totals = Counter()
    image_classes = {}

    for image_record in image_files:
        classes = get_classes_in_image(image_record[2])
        image_classes[image_record] = classes
        class_totals.update(classes)

    target_class_counts = {
        class_id: {
            split: class_totals[class_id] * ratio
            for split, ratio in SPLIT_RATIOS.items()
        }
        for class_id in CLASS_NAMES
    }
    selected_splits = {split: [] for split in OUTPUT_SPLITS}
    assigned_class_counts = {
        split: Counter() for split in OUTPUT_SPLITS
    }
    remaining = list(image_files)
    random.shuffle(remaining)

    # Place multi-class and rarer images first. This makes their class presence
    # influence the assignment before the more flexible single-class images.
    remaining.sort(
        key=lambda record: (
            -len(image_classes[record]),
            sum(1 / max(class_totals[class_id], 1) for class_id in image_classes[record]),
        )
    )

    for image_record in remaining:
        classes = image_classes[image_record]
        available_splits = [
            split
            for split in OUTPUT_SPLITS
            if len(selected_splits[split]) < split_counts[split]
        ]

        def assignment_score(split):
            normalized_class_deficit = sum(
                max(
                    target_class_counts[class_id][split]
                    - assigned_class_counts[split][class_id],
                    0,
                )
                / max(target_class_counts[class_id][split], 1)
                for class_id in classes
            )
            remaining_capacity = (
                split_counts[split] - len(selected_splits[split])
            ) / split_counts[split]
            return normalized_class_deficit, remaining_capacity

        best_split = max(available_splits, key=assignment_score)
        selected_splits[best_split].append(image_record)
        assigned_class_counts[best_split].update(classes)

    return selected_splits


def copy_image_and_label(image_record, output_split):
    """Copy an image and its label into an output split."""

    _, image_path, label_path = image_record
    output_images_dir = OUTPUT_DIR / output_split / "images"
    output_labels_dir = OUTPUT_DIR / output_split / "labels"

    shutil.copy2(image_path, output_images_dir / image_path.name)

    if label_path.exists():
        shutil.copy2(label_path, output_labels_dir / label_path.name)


def get_statistics(image_records):
    """Return image-level and object-level class counts."""

    image_counts = Counter()
    object_counts = Counter()

    for _, _, label_path in image_records:
        classes = get_classes_in_image(label_path)

        for class_id in classes:
            image_counts[class_id] += 1

        if label_path.exists():
            with label_path.open() as label_file:
                for line in label_file:
                    values = line.strip().split()

                    if len(values) == 5:
                        object_counts[int(values[0])] += 1

    return image_counts, object_counts


def print_statistics(split_name, image_records):
    """Print class statistics for an output split."""

    image_counts, object_counts = get_statistics(image_records)

    print(f"\n{split_name.upper()} SET")
    print("=" * 70)
    print(f"Images: {len(image_records)}")

    print("\nImages containing each class:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_name:<10}: {image_counts[class_id]}")

    print("\nObjects in set:")
    total_objects = 0
    for class_id, class_name in CLASS_NAMES.items():
        count = object_counts[class_id]
        total_objects += count
        print(f"  {class_name:<10}: {count}")

    print(f"  {'Total':<10}: {total_objects}")


# ===============================================================================
# Main
# ===============================================================================


def main():
    random.seed(RANDOM_SEED)

    image_files = collect_image_files()
    print(f"Total source images found: {len(image_files)}")

    print("\nConstructing balanced splits from all pooled images:")
    selected_records = create_balanced_splits(image_files)

    for split in OUTPUT_SPLITS:
        output_split_dir = OUTPUT_DIR / split
        if output_split_dir.exists():
            shutil.rmtree(output_split_dir)

        for directory in (
            OUTPUT_DIR / split / "images",
            OUTPUT_DIR / split / "labels",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        for image_record in selected_records[split]:
            copy_image_and_label(image_record, split)

    for split in OUTPUT_SPLITS:
        print_statistics(split, selected_records[split])

    selected_count = sum(len(records) for records in selected_records.values())
    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"Source images considered: {len(image_files)}")
    print(f"Balanced images copied:   {selected_count}")
    print(f"Output directory:          {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
