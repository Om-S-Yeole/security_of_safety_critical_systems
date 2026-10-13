import math
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


class YOLOv8Detector:
    """Reusable inference wrapper for a trained YOLOv8 detector."""

    def __init__(
        self,
        model_path: str,
        image_size: int = 640,
        confidence_threshold: float = 0.60,
        iou_threshold: float = 0.70,
        device: str = "cuda:0",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {self.model_path}")

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")

        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        # Validate the requested device.
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable. "
                "Check your NVIDIA driver and PyTorch installation."
            )

        self.device = device

        # Load the trained Ultralytics YOLO checkpoint.
        self.model = YOLO(str(self.model_path))

        self.max_stride = int(self.model.model.stride.max().item())
        self.image_size = math.ceil(image_size / self.max_stride) * self.max_stride

        # Put the underlying PyTorch model into evaluation mode.
        self.model.model.eval()

        # Freeze all model parameters.
        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)

        print("YOLOv8 detector loaded successfully.")
        print(f"Checkpoint : {self.model_path}")
        print(f"Device     : {self.device}")
        if self.image_size != image_size:
            print(
                f"Image size : {self.image_size} "
                f"(adjusted from {image_size} for stride {self.max_stride})"
            )
        else:
            print(f"Image size : {self.image_size}")
        print(f"Confidence : {self.confidence_threshold}")
        print(f"IoU        : {self.iou_threshold}")

    @torch.inference_mode()
    def predict(
        self,
        source: Any,
        confidence_threshold: float | None = None,
    ) -> list[dict]:
        """
        Run inference on an image, image path, or supported image batch.

        Returns a list containing one dictionary per input image.
        Bounding boxes use xyxy pixel coordinates.
        """

        conf = (
            self.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )

        if not 0.0 <= conf <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        results = self.model.predict(
            source=source,
            imgsz=self.image_size,
            conf=conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        predictions = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                predictions.append(
                    {
                        "image_path": (
                            str(result.path) if result.path is not None else None
                        ),
                        "boxes": [],
                        "scores": [],
                        "class_ids": [],
                        "class_names": [],
                    }
                )
                continue

            xyxy = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()

            class_names = [result.names[class_id] for class_id in class_ids]

            predictions.append(
                {
                    "image_path": (
                        str(result.path) if result.path is not None else None
                    ),
                    "boxes": xyxy,
                    "scores": scores,
                    "class_ids": class_ids,
                    "class_names": class_names,
                }
            )

        return predictions
