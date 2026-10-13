from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image


class UniversalPatch(torch.nn.Module):
    """Trainable RGB patch represented by a PyTorch parameter."""

    def __init__(
        self,
        height: int = 128,
        width: int = 128,
        initialization: str = "uniform",
        min_pixel_value: float = 0.0,
        max_pixel_value: float = 1.0,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if height <= 0 or width <= 0:
            raise ValueError("Patch height and width must be positive.")

        if not (0.0 <= min_pixel_value < max_pixel_value <= 1.0):
            raise ValueError(
                "Pixel limits must satisfy 0 <= min_pixel_value < max_pixel_value <= 1."
            )

        self.height = height
        self.width = width
        self.initialization = initialization
        self.min_pixel_value = min_pixel_value
        self.max_pixel_value = max_pixel_value

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device_name = device

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

        # Use a local generator so initialization does not change
        # the global PyTorch random-number generator state.
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        shape = (1, 3, height, width)

        if initialization == "uniform":
            patch = torch.rand(
                shape,
                generator=generator,
                dtype=torch.float32,
            )
            patch = min_pixel_value + patch * (max_pixel_value - min_pixel_value)

        elif initialization == "gray":
            midpoint = (min_pixel_value + max_pixel_value) / 2.0
            patch = torch.full(
                shape,
                midpoint,
                dtype=torch.float32,
            )

        elif initialization == "random_noise":
            patch = torch.rand(
                shape,
                generator=generator,
                dtype=torch.float32,
            )
            patch = min_pixel_value + patch * (max_pixel_value - min_pixel_value)

        else:
            raise ValueError(
                f"Unknown initialization method: {initialization}. "
                "Choose 'uniform', 'gray', or 'random_noise'."
            )

        # The patch is the trainable parameter.
        self.pixels = torch.nn.Parameter(patch.to(device=device))

    def forward(self) -> torch.Tensor:
        """
        Return the patch tensor.

        Shape: [1, 3, H, W]
        Range: [min_pixel_value, max_pixel_value]

        Clamping is applied to the returned values without
        overwriting the underlying trainable parameter.
        """
        return self.pixels.clamp(
            min=self.min_pixel_value,
            max=self.max_pixel_value,
        )

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """
        Project stored parameters back into the valid pixel range.

        Call after an optimizer update to maintain valid RGB values.
        """
        self.pixels.clamp_(
            min=self.min_pixel_value,
            max=self.max_pixel_value,
        )

    @torch.no_grad()
    def save_image(self, output_path: str) -> Path:
        """Export the current patch as an RGB PNG image."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image_tensor = self.forward().detach().cpu().squeeze(0)
        image = to_pil_image(image_tensor)
        image.save(output_path, format="PNG")

        return output_path

    def save_checkpoint(self, output_path: str) -> Path:
        """Save patch parameters and metadata for later restoration."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "height": self.height,
            "width": self.width,
            "initialization": self.initialization,
            "min_pixel_value": self.min_pixel_value,
            "max_pixel_value": self.max_pixel_value,
        }

        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        device: str | None = None,
    ) -> "UniversalPatch":
        """Restore a patch checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Patch checkpoint not found: {checkpoint_path}")

        # Only load checkpoints from trusted sources.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        patch = cls(
            height=checkpoint["height"],
            width=checkpoint["width"],
            initialization=checkpoint["initialization"],
            min_pixel_value=checkpoint["min_pixel_value"],
            max_pixel_value=checkpoint["max_pixel_value"],
            device=device,
        )

        patch.load_state_dict(checkpoint["state_dict"])
        return patch
