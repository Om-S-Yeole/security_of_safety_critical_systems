"""Apply a trained universal patch to the test split and evaluate YOLO."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.attacker.scripts.dataset import create_dataloader
from src.attacker.scripts.detector import YOLOv8Detector
from src.attacker.scripts.eot import EOTConfig, ImageEOT
from src.attacker.scripts.patch import UniversalPatch
from src.attacker.scripts.train_patch import (
    evaluate_attack_success_rate,
    paste_patch_differentiable,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY_ROOT / "src/attacker/configs/attack.yaml"
DEFAULT_PATCH = REPOSITORY_ROOT / "attacker/weights/patch_final.pt"
DEFAULT_DETECTOR = REPOSITORY_ROOT / "detector/weights/best.pt"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "attacker/experiments/test_patch"


def _resolve(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _make_eot(config: dict, seed: int) -> ImageEOT:
    eot = config.get("eot", {})
    return ImageEOT(
        config=EOTConfig(
            rotation_enabled=eot.get("rotation", {}).get("enabled", True),
            rotation_min_degrees=eot.get("rotation", {}).get("min_degrees", -30.0),
            rotation_max_degrees=eot.get("rotation", {}).get("max_degrees", 30.0),
            scaling_enabled=eot.get("scaling", {}).get("enabled", True),
            scale_min=eot.get("scaling", {}).get("min_scale", 0.75),
            scale_max=eot.get("scaling", {}).get("max_scale", 1.25),
            brightness_enabled=eot.get("brightness", {}).get("enabled", True),
            brightness_min=eot.get("brightness", {}).get("min_factor", 0.70),
            brightness_max=eot.get("brightness", {}).get("max_factor", 1.30),
            perspective_enabled=eot.get("perspective", {}).get("enabled", True),
            perspective_distortion=eot.get("perspective", {}).get(
                "distortion_scale", 0.10
            ),
        ),
        seed=seed,
    )


def _write_dataset_yaml(output_root: Path, class_names: list[str]) -> Path:
    dataset_yaml = output_root / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images",
                "val": "images",
                "names": class_names,
            },
            file,
            sort_keys=False,
        )
    return dataset_yaml


@torch.no_grad()
def export_attacked_test_set(
    patch: UniversalPatch,
    loader,
    eot: ImageEOT,
    output_root: Path,
    source_images: Path,
    patch_coverage: float,
    samples_per_image: int,
    device: str,
    class_names: list[str],
) -> Path:
    """Create attacked PNG images and matching YOLO label files."""
    images_root = output_root / "images"
    labels_root = output_root / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Attacking test images", unit="batch"):
        images = batch["images"].to(device, non_blocking=True)
        boxes = [item.to(device, non_blocking=True) for item in batch["boxes"]]
        attacked = paste_patch_differentiable(
            images,
            patch(),
            boxes,
            eot,
            samples_per_image=samples_per_image,
            patch_coverage=patch_coverage,
        )

        for image, image_path, image_boxes, labels in zip(
            attacked, batch["image_paths"], boxes, batch["labels"]
        ):
            relative_path = Path(image_path).resolve().relative_to(source_images)
            output_image = (images_root / relative_path).with_suffix(".png")
            output_label = labels_root / relative_path.with_suffix(".txt")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)

            to_pil_image(image.cpu().clamp(0.0, 1.0)).save(output_image)
            with output_label.open("w", encoding="utf-8") as file:
                for box, label in zip(image_boxes.cpu().tolist(), labels.tolist()):
                    x1, y1, x2, y2 = box
                    file.write(
                        f"{label} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                        f"{x2 - x1:.6f} {y2 - y1:.6f}\n"
                    )

    return _write_dataset_yaml(output_root, class_names)


def run(args: argparse.Namespace) -> Path:
    with _resolve(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = args.device or config["project"].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {device!r} was requested, but CUDA is unavailable. "
            "Pass --device cpu or configure a CUDA-capable environment."
        )

    detector_config = config["detector"]
    detector = YOLOv8Detector(
        model_path=str(_resolve(args.detector)),
        image_size=detector_config["image_size"],
        confidence_threshold=detector_config["confidence_threshold"],
        iou_threshold=detector_config["iou_threshold"],
        device=device,
    )

    dataset_config = config["dataset"]
    test_images = _resolve(dataset_config["test_images"]).resolve()
    test_labels = _resolve(dataset_config["test_labels"]).resolve()
    loader = create_dataloader(
        images_dir=str(test_images),
        labels_dir=str(test_labels),
        image_size=detector.image_size,
        num_classes=dataset_config["num_classes"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    patch = UniversalPatch.load_checkpoint(str(_resolve(args.patch)), device=device)
    output_root = _resolve(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    patch_config = config["attack"]["patch"]
    eot = _make_eot(config, config["project"].get("seed", 42))
    attacked_yaml = export_attacked_test_set(
        patch=patch,
        loader=loader,
        eot=eot,
        output_root=output_root,
        source_images=test_images,
        patch_coverage=args.patch_coverage
        if args.patch_coverage is not None
        else patch_config.get("coverage", 0.5),
        samples_per_image=args.samples_per_image,
        device=device,
        class_names=dataset_config["class_names"],
    )

    metrics = detector.model.val(
        data=str(attacked_yaml),
        split="val",
        imgsz=detector.image_size,
        device=device,
        project=str(output_root / "ultralytics_validation"),
        name="attacked_test",
        exist_ok=True,
    )
    scalar_metrics = {
        key: float(value)
        for key, value in metrics.results_dict.items()
        if isinstance(value, (int, float))
    }
    attack_success_metrics = evaluate_attack_success_rate(
        detector=detector,
        dataset_yaml_path=attacked_yaml,
        confidence_threshold=config["evaluation"]["confidence_threshold"],
        iou_threshold=config["evaluation"]["iou_threshold"],
    )
    results = {
        "patch": str(_resolve(args.patch)),
        "detector": str(_resolve(args.detector)),
        "attacked_dataset": str(output_root),
        "metrics": scalar_metrics,
        "attack_success_metrics": attack_success_metrics,
    }
    results_path = output_root / "final_results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
    print(f"Attacked test set saved to: {output_root}")
    print(
        f"Attack Success Rate (object-level): "
        f"{attack_success_metrics['attack_success_rate']:.4f}"
    )
    print(
        f"Successful objects: {attack_success_metrics['successful_objects']} / "
        f"{attack_success_metrics['total_objects']}"
    )
    print(
        f"Image Attack Success Rate: "
        f"{attack_success_metrics['image_attack_success_rate']:.4f}"
    )
    print(f"Final detector results saved to: {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply final_patch.pt to the test split and evaluate YOLOv8m."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--patch", default=str(DEFAULT_PATCH))
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-image", type=int, default=1)
    parser.add_argument("--patch-coverage", type=float, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
