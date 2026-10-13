
from pathlib import Path

import torch

from src.attacker.scripts.dataset import create_dataloader

DATA_ROOT = Path("project_data", "balanced_split")

loader = create_dataloader(
    images_dir=str(DATA_ROOT / "train" / "images"),
    labels_dir=str(DATA_ROOT / "train" / "labels"),
    image_size=640,
    num_classes=4,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

print("Image batch shape:", batch["images"].shape)
print("Image dtype:", batch["images"].dtype)
print("Image range:", batch["images"].min().item(),
      batch["images"].max().item())

for i, path in enumerate(batch["image_paths"]):
    print(f"\nImage: {path}")
    print("Boxes:", batch["boxes"][i].shape)
    print("Labels:", batch["labels"][i].tolist())

assert batch["images"].shape[1:] == (3, 640, 640)
assert torch.isfinite(batch["images"]).all()
assert torch.all(batch["images"] >= 0)
assert torch.all(batch["images"] <= 1)

print("\nDataset smoke test passed.")
