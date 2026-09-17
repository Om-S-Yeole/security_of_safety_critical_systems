import random
import shutil
from collections import Counter
from pathlib import Path

# ===============================================================================
# Configuration
# ===============================================================================

TRAIN_IMAGES_DIR = Path("./project_data/train/images")
TRAIN_LABELS_DIR = Path("./project_data/train/labels")

OUTPUT_DIR = Path("./project_data/balanced_split")

VALIDATION_IMAGES_DIR = OUTPUT_DIR / "validation/images"
VALIDATION_LABELS_DIR = OUTPUT_DIR / "validation/labels"

REMAINING_TRAIN_IMAGES_DIR = OUTPUT_DIR / "train/images"
REMAINING_TRAIN_LABELS_DIR = OUTPUT_DIR / "train/labels"

# Number of validation images per class
IMAGES_PER_CLASS = 80

# Reproducible random selection
RANDOM_SEED = 42

# Dataset classes
CLASS_NAMES = {
    0: "Aircraft",
    1: "Building",
    2: "Truck",
    3: "Tank",
}


# ===============================================================================
# Helper functions
# ===============================================================================


def get_image_files():
    """Return all training images."""

    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    return sorted(
        [
            p
            for p in TRAIN_IMAGES_DIR.iterdir()
            if p.is_file() and p.suffix in extensions
        ]
    )


def get_classes_in_image(label_path):
    """Return the set of classes present in an image."""

    classes = set()

    if not label_path.exists():
        return classes

    with open(label_path, "r") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            classes.add(class_id)

    return classes


# ===============================================================================
# Main
# ===============================================================================


def main():

    random.seed(RANDOM_SEED)

    image_files = get_image_files()

    print(f"Total training images found: {len(image_files)}")

    # ---------------------------------------------------------------------------
    # Find single-class images
    # ---------------------------------------------------------------------------

    single_class_images = {class_id: [] for class_id in CLASS_NAMES}

    image_to_classes = {}

    for image_path in image_files:
        label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

        classes = get_classes_in_image(label_path)

        image_to_classes[image_path] = classes

        # We only want images containing exactly ONE class
        if len(classes) == 1:
            class_id = next(iter(classes))

            if class_id in CLASS_NAMES:
                single_class_images[class_id].append(image_path)

    # ---------------------------------------------------------------------------
    # Print available images
    # ---------------------------------------------------------------------------

    print("\nAvailable single-class images:")

    for class_id, class_name in CLASS_NAMES.items():
        count = len(single_class_images[class_id])

        print(f"  {class_name:<10}: {count}")

        if count < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"Not enough single-class {class_name} images. "
                f"Required {IMAGES_PER_CLASS}, found {count}."
            )

    # ---------------------------------------------------------------------------
    # Select exactly 80 images from each class
    # ---------------------------------------------------------------------------

    selected_images = set()

    print(f"\nSelecting {IMAGES_PER_CLASS} images per class...")

    for class_id, class_name in CLASS_NAMES.items():
        candidates = single_class_images[class_id].copy()

        random.shuffle(candidates)

        selected = candidates[:IMAGES_PER_CLASS]

        selected_images.update(selected)

        print(f"  {class_name:<10}: {len(selected)} selected")

    # ---------------------------------------------------------------------------
    # Verify selection
    # ---------------------------------------------------------------------------

    expected_count = len(CLASS_NAMES) * IMAGES_PER_CLASS

    if len(selected_images) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} validation images, "
            f"but selected {len(selected_images)}."
        )

    # ---------------------------------------------------------------------------
    # Create directories
    # ---------------------------------------------------------------------------

    directories = [
        VALIDATION_IMAGES_DIR,
        VALIDATION_LABELS_DIR,
        REMAINING_TRAIN_IMAGES_DIR,
        REMAINING_TRAIN_LABELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Copy images and labels
    # ---------------------------------------------------------------------------

    for image_path in image_files:
        label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

        if image_path in selected_images:
            destination_image = VALIDATION_IMAGES_DIR / image_path.name

            destination_label = VALIDATION_LABELS_DIR / label_path.name

        else:
            destination_image = REMAINING_TRAIN_IMAGES_DIR / image_path.name

            destination_label = REMAINING_TRAIN_LABELS_DIR / label_path.name

        shutil.copy2(image_path, destination_image)

        if label_path.exists():
            shutil.copy2(label_path, destination_label)

    # ---------------------------------------------------------------------------
    # Calculate final statistics
    # ---------------------------------------------------------------------------

    validation_image_counts = Counter()
    validation_object_counts = Counter()

    for image_path in selected_images:
        label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

        classes = get_classes_in_image(label_path)

        # Image-level class count
        for class_id in classes:
            validation_image_counts[class_id] += 1

        # Object-level class count
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    class_id = int(values[0])

                    validation_object_counts[class_id] += 1

    # ---------------------------------------------------------------------------
    # Print final statistics
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BALANCED VALIDATION SET")
    print("=" * 70)

    print(f"Validation images: {len(selected_images)}")

    print("\nImages containing each class:")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_name:<10}: {validation_image_counts[class_id]}")

    print("\nObjects in validation set:")

    total_objects = 0

    for class_id, class_name in CLASS_NAMES.items():
        count = validation_object_counts[class_id]

        total_objects += count

        print(f"  {class_name:<10}: {count}")

    print(f"  {'Total':<10}: {total_objects}")

    # ---------------------------------------------------------------------------
    # Training set statistics
    # ---------------------------------------------------------------------------

    remaining_images = len(image_files) - len(selected_images)

    print("\nRemaining training images:")
    print(f"  {remaining_images}")

    # ---------------------------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"Validation images: {VALIDATION_IMAGES_DIR}")

    print(f"Validation labels: {VALIDATION_LABELS_DIR}")

    print(f"Training images:   {REMAINING_TRAIN_IMAGES_DIR}")

    print(f"Training labels:   {REMAINING_TRAIN_LABELS_DIR}")

    print("=" * 70)


# ===============================================================================
# Run
# ===============================================================================

if __name__ == "__main__":
    main()
