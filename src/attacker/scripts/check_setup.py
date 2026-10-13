from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

# The script is located in <project_root>/attacker/scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "PIL": "Pillow",
    "yaml": "PyYAML",
}


def print_section(title: str) -> None:
    """Print a section heading."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def check_python() -> bool:
    """Check the Python version."""
    print_section("1. Python")

    print(f"Executable : {sys.executable}")
    print(f"Version    : {platform.python_version()}")
    print(f"Platform   : {platform.platform()}")

    supported = sys.version_info >= (3, 9)
    print(f"Status     : {'PASS' if supported else 'WARNING'}")

    if not supported:
        print("Python 3.9 or newer is recommended for this project.")

    return supported


def check_packages() -> bool:
    """Check whether required Python modules can be imported."""
    print_section("2. Required Packages")

    all_available = True

    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version unavailable")
            print(f"[PASS] {package_name}: {version}")
        except Exception as exc:
            all_available = False
            print(f"[FAIL] {package_name}: {exc}")

    return all_available


def check_torch_and_cuda() -> bool:
    """Check PyTorch installation and CUDA/GPU access."""
    print_section("3. PyTorch and NVIDIA GPU")

    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] PyTorch could not be imported: {exc}")
        return False

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA build      : {torch.version.cuda}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[WARNING] PyTorch cannot access CUDA.")
        print("Check your NVIDIA driver and install a PyTorch build")
        print("compatible with your CUDA environment.")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"GPU count       : {gpu_count}")

    for index in range(gpu_count):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / (1024**3)

        print(f"\nGPU {index}")
        print(f"  Name          : {properties.name}")
        print(f"  VRAM          : {memory_gib:.2f} GiB")
        print(f"  Compute cap.  : "
              f"{properties.major}.{properties.minor}")

    # A small tensor operation confirms that basic GPU computation works.
    try:
        test_tensor = torch.ones(4, device="cuda:0")
        result = (test_tensor * 2).sum().item()
        torch.cuda.synchronize()

        if result != 8.0:
            raise RuntimeError(f"Unexpected result: {result}")

        print("\n[PASS] Basic CUDA tensor operation succeeded.")
        return True

    except Exception as exc:
        print(f"\n[FAIL] CUDA computation failed: {exc}")
        return False


def load_config(config_path: Path) -> dict:
    """Load the YAML configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is not installed.") from exc

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise TypeError("The YAML configuration must contain a mapping.")

    return config


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def check_project_files(config_path: Path) -> bool:
    """Check configuration, model weights, and dataset directories."""
    print_section("4. Project Files")

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"[FAIL] Could not load configuration: {exc}")
        return False

    print(f"[PASS] Configuration: {config_path}")

    all_present = True

    detector_config = config.get("detector", {})
    model_path_value = detector_config.get("model_path")

    if model_path_value:
        model_path = resolve_project_path(model_path_value)

        if model_path.is_file():
            print(f"[PASS] Detector weights: {model_path}")
        else:
            print(f"[FAIL] Detector weights not found: {model_path}")
            all_present = False
    else:
        print("[WARNING] detector.model_path is missing.")
        all_present = False

    dataset_config = config.get("dataset", {})
    split_paths = dataset_config.get("splits", {})

    # If your configuration does not define split paths, check the
    # conventional directory structure described below.
    if not split_paths:
        split_paths = {
            split: f"project_data/balanced_split/{split}"
            for split in ("train", "validation", "test")
        }

    for split in ("train", "validation", "test"):
        split_value = split_paths.get(split)

        if not split_value:
            print(f"[WARNING] No path configured for {split} split.")
            continue

        split_path = resolve_project_path(split_value)

        if split_path.is_dir():
            image_dir = split_path / "images"
            label_dir = split_path / "labels"

            if image_dir.is_dir() and label_dir.is_dir():
                print(f"[PASS] {split}: images/ and labels/ found")
            else:
                print(
                    f"[WARNING] {split} directory exists, but expected "
                    f"subdirectories are missing: {split_path}"
                )
        else:
            print(f"[WARNING] Dataset split not found: {split_path}")

    return all_present


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the YOLOv8 project setup."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "src/attacker/configs/attack.yaml",
        help="Path to the project YAML configuration.",
    )
    args = parser.parse_args()

    print("YOLOv8 Project Setup Check")
    print(f"Project root: {PROJECT_ROOT}")

    python_ok = check_python()
    packages_ok = check_packages()
    gpu_ok = check_torch_and_cuda()
    files_ok = check_project_files(args.config.expanduser().resolve())

    print_section("Final Summary")

    checks = {
        "Python": python_ok,
        "Required packages": packages_ok,
        "CUDA/GPU": gpu_ok,
        "Project files": files_ok,
    }

    for name, passed in checks.items():
        print(f"{'[PASS]' if passed else '[FAIL/WARNING]'} {name}")

    if all(checks.values()):
        print("\nAll setup checks passed.")
        return 0

    print("\nSome checks need attention. Review the messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
