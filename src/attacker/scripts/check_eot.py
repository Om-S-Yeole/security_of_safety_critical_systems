from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from src.attacker.scripts.eot import EOTConfig, ImageEOT


def main():
    image_path = Path("project_data/balanced_split/train/images/aircraft-1-_jpg.rf.8a4d3c9f141b580393f093f67ee5b3fb_brightness_1.jpg")

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Replace this example path with a real image: {image_path}"
        )

    image = Image.open(image_path).convert("RGB").resize((640, 640))
    tensor = (
        torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float()
        / 255.0
    )

    config = EOTConfig()
    transform = ImageEOT(config, seed=42)

    output = transform(tensor)

    assert output.shape == tensor.shape
    assert torch.isfinite(output).all()
    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0

    output_path = Path("experiments/eot_test/transformed.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(output).save(output_path)

    print("Input shape:", tuple(tensor.shape))
    print("Output shape:", tuple(output.shape))
    print("Output range:", output.min().item(), output.max().item())
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
