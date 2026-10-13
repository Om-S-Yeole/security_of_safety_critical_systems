import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torch.optim import Adam
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
) -> torch.Tensor:
    """
    Differentiably paste EOT-transformed patches onto target bounding box regions.

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

    Returns
    -------
    patched_images : torch.Tensor
        Batch of images with patches applied, retaining gradient tracking for `patch`.
    """
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

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            for _ in range(samples_per_image):
                # 1. Apply geometric and appearance transformations to patch
                transformed_patch = eot_transform(patch)  # Shape: [1, C, P_H, P_W]

                # 2. Resize patch to fit the target bounding box area
                resized_patch = torch.nn.functional.interpolate(
                    transformed_patch,
                    size=(box_h, box_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # Shape: [C, box_h, box_w]

                # 3. Differentiably blend/paste into target image canvas
                patched_images[b_idx, :, y1:y2, x1:x2] = resized_patch

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
    loss = torch.mean(penalty ** 2)

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
        total_max_target_confidence += target_probabilities.amax(
            dim=(1, 2)
        ).mean().item()
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
            for coordinate, dimension in zip(
                box, (width, height, width, height)
            )
        )
        class_name = (
            class_names[label] if 0 <= label < len(class_names) else str(label)
        )
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        draw.text((x1, max(0, y1 - 18)), f"GT: {class_name}", fill="lime")

    return annotated


def _detector_output_image(detector: YOLOv8Detector, image: torch.Tensor) -> Image.Image:
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
            attacked_path = (
                visualization_dir / f"image_{image_number:03d}_attacked.png"
            )
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


def train_dpatch(config_path: str) -> None:
    # 1. Configuration & Directories Setup
    cfg = load_config(config_path)
    device = cfg["project"].get("device", "cuda:0")
    output_dir = Path(cfg["project"]["output_dir"])
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
        perspective_distortion=eot_cfg_dict.get("perspective", {}).get("distortion_scale", 0.10),
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
        device=device,
    )
    visualization_cfg = cfg.get("evaluation", {}).get("visualization", {})
    visualization_images = save_validation_visualizations(
        detector=detector,
        patch_module=patch_module,
        validation_loader=validation_loader,
        eot_transform=eot_transform,
        samples_per_image=samples_per_image,
        output_dir=output_dir,
        class_names=cfg["dataset"]["class_names"],
        device=device,
        max_images=visualization_cfg.get("num_images", 5),
    )
    final_metrics = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_validation": final_validation,
        "epochs": validation_metrics,
        "validation_visualizations": visualization_images,
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
    parser = argparse.ArgumentParser(description="Train DPatch with EOT against YOLOv8m.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/attacker/configs/attack.yaml",
        help="Path to attack configuration YAML file.",
    )
    args = parser.parse_args()
    train_dpatch(args.config)