from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, RandomPerspective
from torchvision.transforms import functional as TF


@dataclass
class EOTConfig:
    """Configuration for general image transformations."""

    rotation_enabled: bool = True
    rotation_min_degrees: float = -30.0
    rotation_max_degrees: float = 30.0

    scaling_enabled: bool = True
    scale_min: float = 0.75
    scale_max: float = 1.25

    brightness_enabled: bool = True
    brightness_min: float = 0.70
    brightness_max: float = 1.30

    perspective_enabled: bool = True
    perspective_distortion: float = 0.10
    perspective_probability: float = 1.0

    def validate(self) -> None:
        if self.rotation_min_degrees > self.rotation_max_degrees:
            raise ValueError("Invalid rotation range.")

        if not (0.0 < self.scale_min <= self.scale_max):
            raise ValueError("Invalid scaling range.")

        if not (
            0.0 <= self.brightness_min <= self.brightness_max
        ):
            raise ValueError("Invalid brightness range.")

        if not (0.0 <= self.perspective_distortion <= 1.0):
            raise ValueError("Perspective distortion must be in [0, 1].")

        if not (0.0 <= self.perspective_probability <= 1.0):
            raise ValueError("Perspective probability must be in [0, 1].")


class ImageEOT:
    """
    Apply randomly sampled transformations to images.

    Geometric transforms use torchvision operations.
    The output retains the input dimensions.
    """

    def __init__(
        self,
        config: EOTConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or EOTConfig()
        self.config.validate()

        # A local generator makes parameter sampling reproducible.
        self.generator = torch.Generator(device="cpu")

        if seed is not None:
            self.generator.manual_seed(seed)

        self.perspective = RandomPerspective(
            distortion_scale=self.config.perspective_distortion,
            p=self.config.perspective_probability,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

    def _uniform(
        self,
        low: float,
        high: float,
        device: torch.device,
    ) -> float:
        value = torch.rand((), generator=self.generator).item()
        return low + value * (high - low)

    def _transform_one(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError("Expected an image tensor [C, H, W].")

        if not torch.is_floating_point(image):
            raise TypeError("Image tensor must have a floating-point dtype.")

        if not torch.isfinite(image).all():
            raise ValueError("Image contains NaN or infinite values.")

        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError("Expected image values in [0, 1].")

        cfg = self.config
        output = image

        # Rotation.
        if cfg.rotation_enabled:
            angle = self._uniform(
                cfg.rotation_min_degrees,
                cfg.rotation_max_degrees,
                image.device,
            )
            output = TF.rotate(
                output,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

        # Scaling represented by resizing and restoring the original size.
        if cfg.scaling_enabled:
            scale = self._uniform(
                cfg.scale_min,
                cfg.scale_max,
                image.device,
            )

            _, height, width = output.shape

            new_height = max(1, round(height * scale))
            new_width = max(1, round(width * scale))

            resized = TF.resize(
                output,
                [new_height, new_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

            if scale >= 1.0:
                # Center crop when the resized image is larger.
                output = TF.center_crop(resized, [height, width])
            else:
                # Center pad when the resized image is smaller.
                pad_h = height - new_height
                pad_w = width - new_width

                left = pad_w // 2
                right = pad_w - left
                top = pad_h // 2
                bottom = pad_h - top

                output = F.pad(
                    resized,
                    [left, right, top, bottom],
                    mode="constant",
                    value=0.0,
                )

        # Brightness adjustment.
        if cfg.brightness_enabled:
            factor = self._uniform(
                cfg.brightness_min,
                cfg.brightness_max,
                image.device,
            )
            output = TF.adjust_brightness(output, factor)

        # Perspective transformation.
        if cfg.perspective_enabled:
            output = self.perspective(output)

        return output.clamp(0.0, 1.0)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Transform one image or a batch of images.

        Returns the same shape as the input.
        Each image receives independently sampled parameters.
        """
        if images.ndim == 3:
            return self._transform_one(images)

        if images.ndim == 4:
            transformed = [
                self._transform_one(image)
                for image in images
            ]
            return torch.stack(transformed, dim=0)

        raise ValueError(
            "Expected [C, H, W] or [B, C, H, W]."
        )
