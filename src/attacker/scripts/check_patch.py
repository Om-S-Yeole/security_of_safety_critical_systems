
from pathlib import Path

import torch

from src.attacker.scripts.patch import UniversalPatch


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    patch = UniversalPatch(
        height=128,
        width=128,
        initialization="uniform",
        device=device,
        seed=42,
    )

    pixels = patch()

    print("Device:", pixels.device)
    print("Shape:", tuple(pixels.shape))
    print("Minimum:", pixels.min().item())
    print("Maximum:", pixels.max().item())
    print("Trainable:", patch.pixels.requires_grad)

    assert pixels.shape == (1, 3, 128, 128)
    assert pixels.min().item() >= 0.0
    assert pixels.max().item() <= 1.0
    assert patch.pixels.requires_grad

    # Verify that gradients can reach the patch parameter.
    loss = pixels.mean()
    loss.backward()

    assert patch.pixels.grad is not None
    print("Gradient test passed.")

    output_dir = Path("experiments/patch_test")
    patch.save_image(str(output_dir / "patch.png"))
    patch.save_checkpoint(str(output_dir / "patch.pt"))

    restored = UniversalPatch.load_checkpoint(
        str(output_dir / "patch.pt"),
        device=device,
    )

    assert torch.allclose(patch.pixels, restored.pixels)

    print("Save/load test passed.")
    print("Patch test completed successfully.")


if __name__ == "__main__":
    main()
