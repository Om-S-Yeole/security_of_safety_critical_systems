import csv
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from ultralytics import YOLO

from src.helpers import load_yaml

# ===============================================================================
# Paths
# ===============================================================================

TRAINING_YAML_PATH = Path("./src/object_detector/configs/train.yaml")

DATASET_ROOT = Path("./project_data")
TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
TRAIN_LABELS_DIR = DATASET_ROOT / "train" / "labels"

KFOLD_ROOT = Path("./project_data/kfold")
RUNS_ROOT = Path("./runs/kfold")

# Output files
CSV_RESULTS_PATH = RUNS_ROOT / "kfold_results.csv"
TXT_RESULTS_PATH = RUNS_ROOT / "kfold_results.txt"


# ===============================================================================
# K-fold configuration
# ===============================================================================

N_SPLITS = 5
RANDOM_STATE = 42
SHUFFLE = True


# ===============================================================================
# Class information
# ===============================================================================

CLASS_NAMES = {
    0: "Aircraft",
    1: "Building",
    2: "Truck",
    3: "Tank",
}


# ===============================================================================
# Load training configuration
# ===============================================================================

training_yaml = load_yaml(TRAINING_YAML_PATH)
TRAIN_CONFIGS = training_yaml["train_configs"].copy()


# ===============================================================================
# Get image-label pairs
# ===============================================================================


def get_image_label_pairs():

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    pairs = []

    for image_path in sorted(TRAIN_IMAGES_DIR.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue

        label_path = TRAIN_LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for image:\n"
                f"Image: {image_path}\n"
                f"Expected label: {label_path}"
            )

        pairs.append((image_path, label_path))

    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {TRAIN_IMAGES_DIR}")

    return pairs


# ===============================================================================
# Prepare fold dataset
# ===============================================================================


def prepare_fold_dataset(fold_number, train_pairs, val_pairs):

    fold_root = KFOLD_ROOT / f"fold_{fold_number}"

    train_images = fold_root / "train" / "images"
    train_labels = fold_root / "train" / "labels"

    val_images = fold_root / "val" / "images"
    val_labels = fold_root / "val" / "labels"

    # Remove old fold if it exists.
    if fold_root.exists():
        shutil.rmtree(fold_root)

    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    # Copy training data.
    for image_path, label_path in train_pairs:
        shutil.copy2(
            image_path,
            train_images / image_path.name,
        )

        shutil.copy2(
            label_path,
            train_labels / label_path.name,
        )

    # Copy validation data.
    for image_path, label_path in val_pairs:
        shutil.copy2(
            image_path,
            val_images / image_path.name,
        )

        shutil.copy2(
            label_path,
            val_labels / label_path.name,
        )

    return fold_root


# ===============================================================================
# Create YOLO dataset YAML for a fold
# ===============================================================================


def create_fold_yaml(fold_root):

    yaml_path = fold_root / "dataset.yaml"

    yaml_content = f"""
path: {fold_root.resolve()}

train: train/images
val: val/images

names:
  0: Aircraft
  1: Building
  2: Truck
  3: Tank
"""

    yaml_path.write_text(yaml_content.strip() + "\n")

    return yaml_path


# ===============================================================================
# Train and validate one fold
# ===============================================================================


def train_and_validate_fold(fold_number, dataset_yaml):

    print("\n" + "=" * 70)
    print(f"FOLD {fold_number}")
    print("=" * 70)

    # Start a completely fresh YOLOv8m model.
    model = YOLO("yolov8m.pt")

    fold_run_name = f"fold_{fold_number}"

    train_configs = TRAIN_CONFIGS.copy()

    # Use this fold's dataset.
    train_configs["data"] = str(dataset_yaml)

    # Save this fold in its own directory.
    train_configs["project"] = str(RUNS_ROOT)
    train_configs["name"] = fold_run_name

    # Train.
    model.train(**train_configs)

    # ---------------------------------------------------------------------------
    # Locate best weights
    # ---------------------------------------------------------------------------

    best_weights = RUNS_ROOT / fold_run_name / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Best weights were not found for Fold {fold_number}:\n{best_weights}"
        )

    # ---------------------------------------------------------------------------
    # Validate best model
    # ---------------------------------------------------------------------------

    best_model = YOLO(str(best_weights))

    metrics = best_model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=TRAIN_CONFIGS.get("imgsz", 640),
        device=TRAIN_CONFIGS.get("device", 0),
        workers=TRAIN_CONFIGS.get("workers", 8),
        project=str(RUNS_ROOT),
        name=f"{fold_run_name}_validation",
    )

    # Extract overall detection metrics.
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    results = {
        "fold": fold_number,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }

    print(f"\nFold {fold_number} Results:")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  mAP50     : {map50:.4f}")
    print(f"  mAP50-95  : {map50_95:.4f}")

    return results


# ===============================================================================
# Save results to CSV
# ===============================================================================


def save_csv_results(all_results):

    CSV_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "Fold",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
    ]

    with open(
        CSV_RESULTS_PATH,
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        # Fold results.
        for result in all_results:
            writer.writerow(
                {
                    "Fold": result["fold"],
                    "Precision": f"{result['precision']:.6f}",
                    "Recall": f"{result['recall']:.6f}",
                    "mAP50": f"{result['mAP50']:.6f}",
                    "mAP50-95": f"{result['mAP50-95']:.6f}",
                }
            )

        # Mean and standard deviation.
        for label, statistic in [
            ("Mean", np.mean),
            ("Std", lambda x: np.std(x, ddof=1)),
        ]:
            writer.writerow(
                {
                    "Fold": label,
                    "Precision": f"{statistic([r['precision'] for r in all_results]):.6f}",
                    "Recall": f"{statistic([r['recall'] for r in all_results]):.6f}",
                    "mAP50": f"{statistic([r['mAP50'] for r in all_results]):.6f}",
                    "mAP50-95": f"{statistic([r['mAP50-95'] for r in all_results]):.6f}",
                }
            )

    print("\nCSV results saved to:")
    print(f"  {CSV_RESULTS_PATH}")


# ===============================================================================
# Save results to TXT
# ===============================================================================


def save_txt_results(all_results):

    TXT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        TXT_RESULTS_PATH,
        "w",
    ) as txt_file:
        txt_file.write("YOLOv8m 5-Fold Cross-Validation Results\n")

        txt_file.write("=" * 70 + "\n\n")

        txt_file.write(f"Number of folds : {N_SPLITS}\n")

        txt_file.write(f"Random seed     : {RANDOM_STATE}\n")

        txt_file.write(f"Epochs          : {TRAIN_CONFIGS.get('epochs')}\n")

        txt_file.write(f"Image size      : {TRAIN_CONFIGS.get('imgsz')}\n\n")

        # -----------------------------------------------------------------------
        # Individual fold results
        # -----------------------------------------------------------------------

        for result in all_results:
            txt_file.write(f"Fold {result['fold']}\n")

            txt_file.write("-" * 30 + "\n")

            txt_file.write(f"Precision : {result['precision']:.6f}\n")

            txt_file.write(f"Recall    : {result['recall']:.6f}\n")

            txt_file.write(f"mAP50     : {result['mAP50']:.6f}\n")

            txt_file.write(f"mAP50-95  : {result['mAP50-95']:.6f}\n\n")

        # -----------------------------------------------------------------------
        # Overall statistics
        # -----------------------------------------------------------------------

        txt_file.write("Overall Results\n")

        txt_file.write("=" * 30 + "\n")

        for metric_name, key in [
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
        ]:
            values = np.array([result[key] for result in all_results])

            mean = values.mean()
            std = values.std(ddof=1)

            txt_file.write(f"{metric_name:<10}: {mean:.6f} ± {std:.6f}\n")

    print("TXT results saved to:")
    print(f"  {TXT_RESULTS_PATH}")


# ===============================================================================
# Print results to console
# ===============================================================================


def print_summary(all_results):

    print("\n")
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(f"{'Fold':<10}{'Precision':<15}{'Recall':<15}{'mAP50':<15}{'mAP50-95':<15}")

    print("-" * 70)

    for result in all_results:
        print(
            f"{result['fold']:<10}"
            f"{result['precision']:<15.4f}"
            f"{result['recall']:<15.4f}"
            f"{result['mAP50']:<15.4f}"
            f"{result['mAP50-95']:<15.4f}"
        )

    print("-" * 70)

    for metric_name, key in [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("mAP50", "mAP50"),
        ("mAP50-95", "mAP50-95"),
    ]:
        values = np.array([result[key] for result in all_results])

        print(f"{metric_name:<12}: {values.mean():.4f} ± {values.std(ddof=1):.4f}")

    print("=" * 70)


# ===============================================================================
# Main
# ===============================================================================


def main():

    print("=" * 70)
    print("YOLOv8m 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    print(f"Training images : {TRAIN_IMAGES_DIR}")
    print(f"Training labels : {TRAIN_LABELS_DIR}")
    print(f"Number of folds : {N_SPLITS}")
    print(f"Random seed     : {RANDOM_STATE}")

    # ---------------------------------------------------------------------------
    # Get image-label pairs
    # ---------------------------------------------------------------------------

    pairs = get_image_label_pairs()

    print(f"\nTotal training images: {len(pairs)}")

    # ---------------------------------------------------------------------------
    # Create K-fold splitter
    # ---------------------------------------------------------------------------

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE,
    )

    all_results = []

    # ---------------------------------------------------------------------------
    # Run every fold
    # ---------------------------------------------------------------------------

    for fold_number, (train_indices, val_indices) in enumerate(
        kfold.split(pairs),
        start=1,
    ):
        train_pairs = [pairs[i] for i in train_indices]

        val_pairs = [pairs[i] for i in val_indices]

        print("\n" + "-" * 70)
        print(f"Preparing Fold {fold_number}")
        print(f"Training images   : {len(train_pairs)}")
        print(f"Validation images : {len(val_pairs)}")
        print("-" * 70)

        fold_root = prepare_fold_dataset(
            fold_number,
            train_pairs,
            val_pairs,
        )

        dataset_yaml = create_fold_yaml(fold_root)

        results = train_and_validate_fold(
            fold_number,
            dataset_yaml,
        )

        all_results.append(results)

    # ---------------------------------------------------------------------------
    # Print and save results
    # ---------------------------------------------------------------------------

    print_summary(all_results)

    save_csv_results(all_results)

    save_txt_results(all_results)

    print("\nK-fold cross-validation complete.")


# ===============================================================================
# Run
# ===============================================================================

if __name__ == "__main__":
    main()
