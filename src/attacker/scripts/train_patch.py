import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch.optim import Adam
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.attacker.scripts.dataset import create_dataloader
from src.attacker.scripts.detector import YOLOv8Detector
from src.attacker.scripts.eot import EOTConfig, ImageEOT
from src.attacker.scripts.patch import UniversalPatch


def load_config(config_path: str) -> dict:
    """Load attack configuration YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def paste_patch_differentiable(
    images: torch.Tensor,
    patch: torch.Tensor,
    boxes_list: list[torch.Tensor],
    eot_transform: ImageEOT,
    samples_per_image: int = 1,
    patch_coverage: float = 0.5,
) -> torch.Tensor:
    """
    Differentiably paste EOT-transformed patches onto target bounding boxes.

    The patch is scaled to fit within ``patch_coverage`` of both bounding-box
    dimensions and is randomly positioned within the box. This keeps the
    remainder of each object visible while making the patch size object-aware.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images [B, C, H, W] in range [0, 1].
    patch : torch.Tensor
        Patch tensor [1, C, P_H, P_W].
    boxes_list : list[torch.Tensor]
        List of box coordinate tensors [N, 4] (normalized xyxy) per image.
    eot_transform : ImageEOT
        EOT augmentation engine.
    samples_per_image : int
        Number of augmented patch variations sampled per bounding box.
    patch_coverage : float
        Maximum fraction of each bounding-box dimension occupied by the patch.
        Must be in ``(0, 1]``.

    Returns
    -------
    patched_images : torch.Tensor
        Batch of images with patches applied, retaining gradient tracking for `patch`.
    """
    if not 0.0 < patch_coverage <= 1.0:
        raise ValueError("patch_coverage must be in the range (0, 1].")

    batch_size, _, img_h, img_w = images.shape
    patched_images = images.clone()

    for b_idx in range(batch_size):
        boxes = boxes_list[b_idx]
        if boxes.numel() == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = (
                int(value * image_size)
                for value, image_size in zip(
                    box.detach().cpu().tolist(),
                    (img_w, img_h, img_w, img_h),
                )
            )

            box_x1 = max(0, min(img_w, x1))
            box_y1 = max(0, min(img_h, y1))
            box_x2 = max(0, min(img_w, x2))
            box_y2 = max(0, min(img_h, y2))
            box_w = max(1, box_x2 - box_x1)
            box_h = max(1, box_y2 - box_y1)

            patch_height, patch_width = patch.shape[-2:]
            scale = min(
                box_w * patch_coverage / patch_width,
                box_h * patch_coverage / patch_height,
            )
            patch_w = max(1, min(box_w, round(patch_width * scale)))
            patch_h = max(1, min(box_h, round(patch_height * scale)))

            for _ in range(samples_per_image):
                # 1. Apply geometric and appearance transformations to patch
                transformed_patch = eot_transform(patch)  # Shape: [1, C, P_H, P_W]

                # 2. Resize patch to a fraction of the target box, preserving
                # its aspect ratio rather than covering the whole object.
                resized_patch = torch.nn.functional.interpolate(
                    transformed_patch,
                    size=(patch_h, patch_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # Shape: [C, patch_h, patch_w]

                # 3. Randomly place the patch fully inside the object box.
                max_x_offset = box_w - patch_w
                max_y_offset = box_h - patch_h
                x_offset = (
                    int(torch.randint(max_x_offset + 1, (), device=images.device))
                    if max_x_offset
                    else 0
                )
                y_offset = (
                    int(torch.randint(max_y_offset + 1, (), device=images.device))
                    if max_y_offset
                    else 0
                )
                patch_x1 = box_x1 + x_offset
                patch_y1 = box_y1 + y_offset

                # Differentiably blend/paste into target image canvas.
                patched_images[
                    b_idx,
                    :,
                    patch_y1 : patch_y1 + patch_h,
                    patch_x1 : patch_x1 + patch_w,
                ] = resized_patch

    return patched_images


def compute_object_disappearance_loss(
    model_predictions: torch.Tensor,
    target_classes: list[int],
    confidence_threshold: float = 0.60,
) -> torch.Tensor:
    """
    Calculate loss aimed at suppressing object detection confidence scores below 0.60.

    Parameters
    ----------
    model_predictions : torch.Tensor
        Raw output tensor from YOLOv8 head [B, 4 + num_classes, num_anchors].
    target_classes : list[int]
        Class IDs to suppress (e.g., [0, 1, 2, 3]).
    confidence_threshold : float
        Desired upper bound for detection scores.

    Returns
    -------
    loss : torch.Tensor
        Scalar differentiable loss tensor.
    """
    # YOLOv8 predictions tensor shape: [Batch, 4 + Num_Classes, Num_Anchors]
    # Class logits or probabilities start after the 4 box coordinate values.
    class_logits = model_predictions[:, 4:, :]  # [B, Num_Classes, Anchors]
    target_logits = class_logits[:, target_classes, :]  # Filter target classes

    # Compute probabilities using sigmoid
    probs = torch.sigmoid(target_logits)

    # Penalize scores that exceed the confidence threshold (hinge loss formulation)
    penalty = torch.relu(probs - confidence_threshold)
    loss = torch.mean(penalty**2)

    return loss


@torch.no_grad()
def evaluate_patch(
    model: torch.nn.Module,
    patch_module: UniversalPatch,
    data_loader,
    target_classes: list[int],
    confidence_threshold: float,
    eot_transform: ImageEOT,
    samples_per_image: int,
    device: str,
    patch_coverage: float = 0.5,
) -> dict[str, float]:
    """Evaluate a patch on a dataset without updating its parameters."""
    total_loss = 0.0
    total_target_confidence = 0.0
    total_max_target_confidence = 0.0
    batch_count = 0

    for batch in data_loader:
        images = batch["images"].to(device, non_blocking=True)
        boxes_list = [b.to(device, non_blocking=True) for b in batch["boxes"]]
        patched_images = paste_patch_differentiable(
            images=images,
            patch=patch_module(),
            boxes_list=boxes_list,
            eot_transform=eot_transform,
            samples_per_image=samples_per_image,
            patch_coverage=patch_coverage,
        )

        raw_predictions = model(patched_images)
        if isinstance(raw_predictions, (tuple, list)):
            raw_predictions = raw_predictions[0]

        loss = compute_object_disappearance_loss(
            model_predictions=raw_predictions,
            target_classes=target_classes,
            confidence_threshold=confidence_threshold,
        )
        target_probabilities = torch.sigmoid(
            raw_predictions[:, 4:, :][:, target_classes, :]
        )

        total_loss += loss.item()
        total_target_confidence += target_probabilities.mean().item()
        total_max_target_confidence += (
            target_probabilities.amax(dim=(1, 2)).mean().item()
        )
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("Validation loader is empty.")

    return {
        "loss": total_loss / batch_count,
        "mean_target_confidence": total_target_confidence / batch_count,
        "mean_max_target_confidence": total_max_target_confidence / batch_count,
    }


def _draw_ground_truth(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
) -> Image.Image:
    """Draw ground-truth boxes and class names on a copy of an image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for box, label in zip(boxes.tolist(), labels.tolist()):
        x1, y1, x2, y2 = (
            coordinate * dimension
            for coordinate, dimension in zip(box, (width, height, width, height))
        )
        class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1, max(0, y1 - 18)), f"GT: {class_name}", fill="lime")

    return annotated


def _detector_output_image(
    detector: YOLOv8Detector, image: torch.Tensor
) -> Image.Image:
    """Run detector inference and convert Ultralytics annotations to a PIL image."""
    results = detector.model.predict(
        source=image.unsqueeze(0),
        imgsz=detector.image_size,
        conf=detector.confidence_threshold,
        iou=detector.iou_threshold,
        device=detector.device,
        verbose=False,
    )
    plotted = results[0].plot()
    return Image.fromarray(plotted[:, :, ::-1])


@torch.no_grad()
def save_validation_visualizations(
    detector: YOLOv8Detector,
    patch_module: UniversalPatch,
    validation_loader,
    eot_transform: ImageEOT,
    samples_per_image: int,
    output_dir: Path,
    class_names: list[str],
    device: str,
    max_images: int = 5,
    patch_coverage: float = 0.5,
) -> list[dict[str, str]]:
    """Save clean and attacked detector outputs for validation examples."""
    if max_images <= 0:
        return []

    visualization_dir = output_dir / "validation_visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    saved_images: list[dict[str, str]] = []

    for batch in validation_loader:
        images = batch["images"].to(device, non_blocking=True)
        boxes_list = [b.to(device, non_blocking=True) for b in batch["boxes"]]
        labels_list = batch["labels"]
        patched_images = paste_patch_differentiable(
            images=images,
            patch=patch_module(),
            boxes_list=boxes_list,
            eot_transform=eot_transform,
            samples_per_image=samples_per_image,
            patch_coverage=patch_coverage,
        )

        for index, (image, patched_image, boxes, labels) in enumerate(
            zip(images, patched_images, boxes_list, labels_list)
        ):
            if boxes.numel() == 0:
                continue

            image_number = len(saved_images) + 1
            clean_detector_image = _detector_output_image(detector, image)
            clean_image = _draw_ground_truth(
                clean_detector_image,
                boxes.cpu(),
                labels,
                class_names,
            )
            attacked_image = _detector_output_image(detector, patched_image)

            clean_path = visualization_dir / f"image_{image_number:03d}_clean.png"
            attacked_path = visualization_dir / f"image_{image_number:03d}_attacked.png"
            clean_image.save(clean_path)
            attacked_image.save(attacked_path)
            saved_images.append(
                {
                    "source_image": batch["image_paths"][index],
                    "clean_output": str(clean_path),
                    "attacked_output": str(attacked_path),
                }
            )

            if len(saved_images) >= max_images:
                return saved_images

    return saved_images


@torch.no_grad()
def export_attacked_validation_set(
    patch_module: UniversalPatch,
    validation_loader,
    eot_transform: ImageEOT,
    samples_per_image: int,
    patch_coverage: float,
    source_images_dir: str,
    source_labels_dir: str,
    output_dir: Path,
    class_names: list[str],
    device: str,
) -> Path:
    """Export every attacked validation image and its YOLO labels."""
    attacked_root = output_dir / "attacked_validation"
    images_dir = attacked_root / "images"
    labels_dir = attacked_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    source_images_root = Path(source_images_dir).resolve()
    source_labels_root = Path(source_labels_dir).resolve()

    for batch in validation_loader:
        images = batch["images"].to(device, non_blocking=True)
        boxes_list = [b.to(device, non_blocking=True) for b in batch["boxes"]]
        patched_images = paste_patch_differentiable(
            images=images,
            patch=patch_module(),
            boxes_list=boxes_list,
            eot_transform=eot_transform,
            samples_per_image=samples_per_image,
            patch_coverage=patch_coverage,
        )

        for index, (patched_image, boxes, labels) in enumerate(
            zip(patched_images, boxes_list, batch["labels"])
        ):
            source_image_path = Path(batch["image_paths"][index]).resolve()
            relative_path = source_image_path.relative_to(source_images_root)
            output_image_path = (images_dir / relative_path).with_suffix(".png")
            output_label_path = labels_dir / relative_path.with_suffix(".txt")
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            output_label_path.parent.mkdir(parents=True, exist_ok=True)

            to_pil_image(patched_image.cpu().clamp(0.0, 1.0)).save(output_image_path)

            source_label_path = source_labels_root / relative_path.with_suffix(".txt")
            if not source_label_path.is_file():
                raise FileNotFoundError(
                    f"Validation label file not found: {source_label_path}"
                )

            with output_label_path.open("w", encoding="utf-8") as label_file:
                for box, label in zip(boxes.cpu().tolist(), labels.tolist()):
                    x1, y1, x2, y2 = box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    x_center = x1 + box_width / 2.0
                    y_center = y1 + box_height / 2.0
                    label_file.write(
                        f"{label} {x_center:.6f} {y_center:.6f} "
                        f"{box_width:.6f} {box_height:.6f}\n"
                    )

    dataset_yaml_path = attacked_root / "dataset.yaml"
    dataset_config = {
        "path": str(attacked_root.resolve()),
        "train": "images",
        "val": "images",
        "names": class_names,
    }
    with dataset_yaml_path.open("w", encoding="utf-8") as dataset_file:
        yaml.safe_dump(dataset_config, dataset_file, sort_keys=False)

    return dataset_yaml_path


def validate_attacked_validation_set(
    detector: YOLOv8Detector,
    dataset_yaml_path: Path,
    output_dir: Path,
) -> dict[str, float]:
    """Run Ultralytics' native validation and return its scalar metrics."""
    metrics = detector.model.val(
        data=str(dataset_yaml_path),
        split="val",
        imgsz=detector.image_size,
        device=detector.device,
        project=str(output_dir / "ultralytics_validation"),
        name="attacked",
        exist_ok=True,
    )
    return {
        key: float(value)
        for key, value in metrics.results_dict.items()
        if isinstance(value, (int, float))
    }


def _box_iou_xyxy(
    first: tuple[float, float, float, float],
    second: list[float],
) -> float:
    """Calculate IoU between two boxes in xyxy pixel coordinates."""
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_attack_success_rate(
    detector: YOLOv8Detector,
    dataset_yaml_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    """
    Evaluate object-level attack success on exported attacked images.

    A ground-truth object is counted as successfully attacked when no
    same-class prediction has both confidence >= ``confidence_threshold`` and
    IoU >= ``iou_threshold``. Therefore, missed, misclassified, and
    low-confidence detections count as successful attacks.
    """
    dataset_config = load_config(str(dataset_yaml_path))
    dataset_root = Path(dataset_config["path"])
    images_root = dataset_root / dataset_config["val"]
    labels_root = dataset_root / "labels"
    image_paths = sorted(path for path in images_root.rglob("*") if path.is_file())
    if not image_paths:
        raise RuntimeError(f"No attacked validation images found in {images_root}")

    total_objects = 0
    successful_objects = 0
    successful_images = 0

    if detector.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    for image_path in tqdm(
        image_paths,
        desc="Evaluating attack success",
        unit="image",
    ):
        # Predict one image at a time. Passing the complete validation list to
        # Ultralytics can create a large temporary batch after model.val().
        predictions = detector.predict(
            source=str(image_path),
            confidence_threshold=0.0,
        )
        if len(predictions) != 1:
            raise RuntimeError(
                f"Detector returned {len(predictions)} predictions for {image_path}."
            )
        prediction = predictions[0]

        with Image.open(image_path) as image:
            image_width, image_height = image.size
        label_path = labels_root / image_path.relative_to(images_root).with_suffix(
            ".txt"
        )
        if not label_path.is_file():
            raise FileNotFoundError(
                f"Attacked validation label not found: {label_path}"
            )

        image_success = True
        with label_path.open("r", encoding="utf-8") as label_file:
            labels = [line.split() for line in label_file if line.strip()]

        for values in labels:
            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO label in {label_path}: {' '.join(values)}"
                )
            class_id = int(values[0])
            x_center, y_center, box_width, box_height = (
                float(value) for value in values[1:]
            )
            ground_truth_box = (
                (x_center - box_width / 2.0) * image_width,
                (y_center - box_height / 2.0) * image_height,
                (x_center + box_width / 2.0) * image_width,
                (y_center + box_height / 2.0) * image_height,
            )
            total_objects += 1

            has_successful_detection = any(
                predicted_class == class_id
                and score >= confidence_threshold
                and _box_iou_xyxy(ground_truth_box, predicted_box) >= iou_threshold
                for predicted_box, score, predicted_class in zip(
                    prediction["boxes"],
                    prediction["scores"],
                    prediction["class_ids"],
                )
            )
            if not has_successful_detection:
                successful_objects += 1
            else:
                image_success = False

        if image_success:
            successful_images += 1

    if total_objects == 0:
        raise RuntimeError("Attacked validation labels contain no objects.")

    return {
        "attack_success_rate": successful_objects / total_objects,
        "successful_objects": successful_objects,
        "total_objects": total_objects,
        "image_attack_success_rate": successful_images / len(image_paths),
        "successful_images": successful_images,
        "total_images": len(image_paths),
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }


def train_dpatch(config_path: str) -> None:
    # 1. Configuration & Directories Setup
    cfg = load_config(config_path)
    device = cfg["project"].get("device", "cuda:0")
    output_dir = Path(cfg["project"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration for experiment tracking
    with (output_dir / "attack_config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # 2. Load Target Object Detector
    detector = YOLOv8Detector(
        model_path=cfg["detector"]["model_path"],
        image_size=cfg["detector"]["image_size"],
        confidence_threshold=cfg["detector"]["confidence_threshold"],
        iou_threshold=cfg["detector"]["iou_threshold"],
        device=device,
    )
    yolo_model = detector.model.model.to(device)
    yolo_model.eval()  # Freeze detector during patch optimization[cite: 2, 6]

    # The raw model forward requires dimensions divisible by its maximum stride.
    # Use the detector's effective size for the dataset as well as the model.
    train_loader = create_dataloader(
        images_dir=cfg["dataset"]["train_images"],
        labels_dir=cfg["dataset"]["train_labels"],
        image_size=detector.image_size,
        num_classes=cfg["dataset"]["num_classes"],
        batch_size=cfg["optimization"]["batch_size"],
        shuffle=cfg["dataset"].get("shuffle", True),
        num_workers=cfg["dataset"].get("num_workers", 4),
    )
    validation_loader = create_dataloader(
        images_dir=cfg["dataset"]["val_images"],
        labels_dir=cfg["dataset"]["val_labels"],
        image_size=detector.image_size,
        num_classes=cfg["dataset"]["num_classes"],
        batch_size=cfg["optimization"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"].get("num_workers", 4),
    )

    # 3. Initialize Patch Module[cite: 5, 6]
    patch_cfg = cfg["attack"]["patch"]
    patch_module = UniversalPatch(
        height=patch_cfg["height"],
        width=patch_cfg["width"],
        initialization=patch_cfg.get("initialization", "uniform"),
        min_pixel_value=patch_cfg.get("min_pixel_value", 0.0),
        max_pixel_value=patch_cfg.get("max_pixel_value", 1.0),
        device=device,
        seed=cfg["project"].get("seed", 42),
    )
    print(f"Model device     : {next(yolo_model.parameters()).device}")
    print(f"Patch device     : {patch_module.pixels.device}")

    # 5. Initialize EOT Pipeline[cite: 3, 6]
    eot_cfg_dict = cfg.get("eot", {})
    eot_config = EOTConfig(
        rotation_enabled=eot_cfg_dict.get("rotation", {}).get("enabled", True),
        rotation_min_degrees=eot_cfg_dict.get("rotation", {}).get("min_degrees", -30.0),
        rotation_max_degrees=eot_cfg_dict.get("rotation", {}).get("max_degrees", 30.0),
        scaling_enabled=eot_cfg_dict.get("scaling", {}).get("enabled", True),
        scale_min=eot_cfg_dict.get("scaling", {}).get("min_scale", 0.75),
        scale_max=eot_cfg_dict.get("scaling", {}).get("max_scale", 1.25),
        brightness_enabled=eot_cfg_dict.get("brightness", {}).get("enabled", True),
        brightness_min=eot_cfg_dict.get("brightness", {}).get("min_factor", 0.70),
        brightness_max=eot_cfg_dict.get("brightness", {}).get("max_factor", 1.30),
        perspective_enabled=eot_cfg_dict.get("perspective", {}).get("enabled", True),
        perspective_distortion=eot_cfg_dict.get("perspective", {}).get(
            "distortion_scale", 0.10
        ),
    )
    eot_transform = ImageEOT(config=eot_config, seed=cfg["project"].get("seed", 42))

    # 6. Optimizer Setup[cite: 6]
    optimizer = Adam(
        [patch_module.pixels],
        lr=cfg["optimization"]["learning_rate"],
    )

    num_epochs = cfg["optimization"]["num_epochs"]
    target_classes = cfg["attack"]["target_classes"]
    conf_thresh = cfg["evaluation"]["confidence_threshold"]
    grad_clip = cfg["optimization"].get("gradient_clip_norm", 1.0)
    samples_per_image = eot_cfg_dict.get("samples_per_image", 1)
    patch_coverage = patch_cfg.get("coverage", 0.5)
    validation_metrics: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_patch_pixels = patch_module.pixels.detach().clone()

    print("\nStarting DPatch + EOT Training Loop...")
    print(f"Target Classes    : {target_classes} ({cfg['dataset']['class_names']})")
    print(f"Epochs            : {num_epochs}")
    print(f"Patch Resolution  : {patch_cfg['height']}x{patch_cfg['width']}")
    print(f"Output Directory  : {output_dir}\n")

    # 7. Training Optimization Loop
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")

        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)
            boxes_list = [b.to(device, non_blocking=True) for b in batch["boxes"]]

            optimizer.zero_grad()

            # Forward pass: extract trainable patch tensor
            current_patch = patch_module()

            # Apply differentiable EOT patch placement
            patched_images = paste_patch_differentiable(
                images=images,
                patch=current_patch,
                boxes_list=boxes_list,
                eot_transform=eot_transform,
                samples_per_image=samples_per_image,
                patch_coverage=patch_coverage,
            )

            # Extract model head predictions
            # For YOLOv8 PyTorch model forward pass returns raw output tensors
            raw_predictions = yolo_model(patched_images)
            if isinstance(raw_predictions, (tuple, list)):
                raw_predictions = raw_predictions[0]

            # Compute object disappearance loss
            loss = compute_object_disappearance_loss(
                model_predictions=raw_predictions,
                target_classes=target_classes,
                confidence_threshold=conf_thresh,
            )

            # Backpropagation
            loss.backward()

            # Gradient clipping safeguard[cite: 6]
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([patch_module.pixels], grad_clip)

            optimizer.step()

            # Project parameters back to valid [0, 1] RGB range
            patch_module.project_parameters_()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.6f}")

        epoch_validation = evaluate_patch(
            model=yolo_model,
            patch_module=patch_module,
            data_loader=validation_loader,
            target_classes=target_classes,
            confidence_threshold=conf_thresh,
            eot_transform=eot_transform,
            samples_per_image=samples_per_image,
            patch_coverage=patch_coverage,
            device=device,
        )
        epoch_validation_record: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": avg_loss,
            **epoch_validation,
        }
        validation_metrics.append(epoch_validation_record)
        print(
            f"Epoch [{epoch}/{num_epochs}] Validation Loss: "
            f"{epoch_validation['loss']:.6f} | "
            f"Mean Target Confidence: "
            f"{epoch_validation['mean_target_confidence']:.6f}"
        )

        if epoch_validation["loss"] < best_validation_loss:
            best_validation_loss = epoch_validation["loss"]
            best_epoch = epoch
            best_patch_pixels = patch_module.pixels.detach().clone()
            print(f"--> New best patch selected at epoch {epoch}.")

        # Save periodic checkpoints and visualizations[cite: 5, 6]
        save_every = cfg["optimization"].get("save_every_epochs", 5)
        if epoch % save_every == 0 or epoch == num_epochs:
            ckpt_path = output_dir / f"patch_epoch_{epoch}.pt"
            img_path = output_dir / f"patch_epoch_{epoch}.png"

            patch_module.save_checkpoint(str(ckpt_path))
            patch_module.save_image(str(img_path))
            print(f"--> Checkpoint saved: {ckpt_path}")

    # Restore the patch with the best validation loss before writing final outputs.
    with torch.no_grad():
        patch_module.pixels.copy_(best_patch_pixels)

    final_ckpt = output_dir / "patch_final.pt"
    final_img = output_dir / "patch_final.png"
    patch_module.save_checkpoint(str(final_ckpt))
    patch_module.save_image(str(final_img))

    final_validation = evaluate_patch(
        model=yolo_model,
        patch_module=patch_module,
        data_loader=validation_loader,
        target_classes=target_classes,
        confidence_threshold=conf_thresh,
        eot_transform=eot_transform,
        samples_per_image=samples_per_image,
        patch_coverage=patch_coverage,
        device=device,
    )
    visualization_cfg = cfg.get("evaluation", {}).get("visualization", {})
    visualization_images = save_validation_visualizations(
        detector=detector,
        patch_module=patch_module,
        validation_loader=validation_loader,
        eot_transform=eot_transform,
        samples_per_image=samples_per_image,
        patch_coverage=patch_coverage,
        output_dir=output_dir,
        class_names=cfg["dataset"]["class_names"],
        device=device,
        max_images=visualization_cfg.get("num_images", 5),
    )
    attacked_dataset_yaml = export_attacked_validation_set(
        patch_module=patch_module,
        validation_loader=validation_loader,
        eot_transform=eot_transform,
        samples_per_image=1,
        patch_coverage=patch_coverage,
        source_images_dir=cfg["dataset"]["val_images"],
        source_labels_dir=cfg["dataset"]["val_labels"],
        output_dir=output_dir,
        class_names=cfg["dataset"]["class_names"],
        device=device,
    )
    attacked_validation_metrics = validate_attacked_validation_set(
        detector=detector,
        dataset_yaml_path=attacked_dataset_yaml,
        output_dir=output_dir,
    )
    attack_success_metrics = evaluate_attack_success_rate(
        detector=detector,
        dataset_yaml_path=attacked_dataset_yaml,
        confidence_threshold=cfg["evaluation"]["confidence_threshold"],
        iou_threshold=cfg["evaluation"]["iou_threshold"],
    )
    print(f"Attacked validation dataset saved to: {attacked_dataset_yaml.parent}")
    print(
        f"Attack Success Rate (object-level): "
        f"{attack_success_metrics['attack_success_rate']:.4f}"
    )
    print(
        f"Successful objects: {attack_success_metrics['successful_objects']} / "
        f"{attack_success_metrics['total_objects']}"
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_validation": final_validation,
        "epochs": validation_metrics,
        "validation_visualizations": visualization_images,
        "attacked_validation_dataset": str(attacked_dataset_yaml.parent),
        "attacked_validation_metrics": attacked_validation_metrics,
        "attack_success_metrics": attack_success_metrics,
    }
    metrics_path = output_dir / "validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(final_metrics, file, indent=4, allow_nan=False)

    print(f"\nTraining Complete! Best patch saved to: {final_img}")
    print(f"Validation metrics saved to: {metrics_path}")
    print(
        f"Validation visualizations saved to: "
        f"{output_dir / 'validation_visualizations'}"
    )
    print(f"Validation image pairs written: {len(visualization_images)}")
    print("\nFinal Best-Patch Validation Metrics")
    print("-" * 38)
    for name, value in final_validation.items():
        print(f"{name:<30}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DPatch with EOT against YOLOv8m."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/attacker/configs/attack.yaml",
        help="Path to attack configuration YAML file.",
    )
    args = parser.parse_args()
    train_dpatch(args.config)
