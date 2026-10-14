import math
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class YOLODetectionDataset(Dataset):
    """
    Loads images and their corresponding YOLO-format annotations.

    Returns a dictionary containing:
        image:       Float tensor [3, H, W], range [0, 1]
        boxes:       Float tensor [N, 4], normalized xyxy coordinates
        labels:      Long tensor [N], class IDs
        image_path:  Original image path
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int | None = 640,
        num_classes: int = 4,
        strict: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.num_classes = num_classes
        self.strict = strict

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {self.labels_dir}")

        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive or None.")

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.image_paths = sorted(
            path
            for path in self.images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(f"No supported images found in {self.images_dir}")

        # Check annotation availability before training/evaluation.
        missing_labels = [
            path for path in self.image_paths if not self._label_path(path).is_file()
        ]

        if missing_labels and strict:
            examples = "\n".join(str(path) for path in missing_labels[:5])
            raise FileNotFoundError(
                f"{len(missing_labels)} images have no label file. "
                "If empty-label images are intentional, set strict=False.\n"
                f"Examples:\n{examples}"
            )

    def _label_path(self, image_path: Path) -> Path:
        """
        Map an image to its label file.

        Nested image folders are supported when the corresponding
        relative directory structure exists under labels_dir.
        """
        relative_path = image_path.relative_to(self.images_dir)
        return (self.labels_dir / relative_path).with_suffix(".txt")

    def _read_labels(self, image_path: Path):
        """
        Read YOLO labels and return normalized xyxy boxes and class IDs.
        """
        label_path = self._label_path(image_path)

        if not label_path.exists():
            # Permitted only when strict=False.
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )

        boxes = []
        labels = []

        with label_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected 5 values, "
                        f"got {len(values)}."
                    )

                try:
                    class_value = float(values[0])
                    coordinates = [float(v) for v in values[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "annotation contains a non-numeric value."
                    ) from exc

                if not class_value.is_integer():
                    raise ValueError(
                        f"{label_path}:{line_number}: class ID must be an integer."
                    )

                class_id = int(class_value)

                if not 0 <= class_id < self.num_classes:
                    raise ValueError(
                        f"{label_path}:{line_number}: class ID {class_id} "
                        f"is outside [0, {self.num_classes - 1}]."
                    )

                x_center, y_center, width, height = coordinates

                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "annotation contains a non-finite value."
                    )

                if width <= 0.0 or height <= 0.0:
                    raise ValueError(
                        f"{label_path}:{line_number}: "
                        "box width and height must be positive."
                    )

                x1 = x_center - width / 2.0
                y1 = y_center - height / 2.0
                x2 = x_center + width / 2.0
                y2 = y_center + height / 2.0

                # An object can legitimately touch or cross an image edge.
                # Clip its visible portion to the normalized image bounds.
                # Ignore objects that are fully outside the transformed image.
                if x2 <= 0.0 or y2 <= 0.0 or x1 >= 1.0 or y1 >= 1.0:
                    continue

                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(1.0, x2)
                y2 = min(1.0, y2)

                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)

        if not boxes:
            return (
                torch.empty((0, 4), dtype=torch.float32),
                torch.empty((0,), dtype=torch.long),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.image_size is not None:
                # Resize to a fixed square resolution.
                # This simple resize changes aspect ratio if needed.
                image = image.resize(
                    (self.image_size, self.image_size),
                    Image.Resampling.BILINEAR,
                )

            image_tensor = TF.to_tensor(image)

        boxes, labels = self._read_labels(image_path)

        return {
            "image": image_tensor,
            "boxes": boxes,
            "labels": labels,
            "image_path": str(image_path),
        }


def detection_collate_fn(batch):
    """
    Collate variable numbers of objects per image.

    Images are stacked; boxes and labels remain lists because
    different images contain different numbers of objects.
    """
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }


def create_dataloader(
    images_dir: str,
    labels_dir: str,
    image_size: int = 640,
    num_classes: int = 4,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    strict: bool = True,
):
    """Create a DataLoader for a YOLODetectionDataset."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    dataset = YOLODetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=image_size,
        num_classes=num_classes,
        strict=strict,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # Do not keep worker processes alive after an iterator is exhausted.
        # This avoids atexit cleanup races when training creates several
        # iterators over the same loaders.
        persistent_workers=False,
        collate_fn=detection_collate_fn,
    )

    return loader
