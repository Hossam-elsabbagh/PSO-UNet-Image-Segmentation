"""Image and trimap transforms shared by all Oxford-IIIT Pet experiments."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def trimap_to_binary(trimap: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map 1=pet, 2=background, 3=ignore to target and valid masks.

    Both returned arrays have shape ``(height, width)`` and dtype ``float32``.
    Values at ignored pixels in the target are zero, but only the valid mask
    determines whether they contribute to losses and metrics.
    """

    labels = np.asarray(trimap)
    if labels.ndim != 2:
        raise ValueError(f"Expected a single-channel trimap, got shape {labels.shape}")
    unexpected = np.setdiff1d(np.unique(labels), np.array([1, 2, 3]))
    if unexpected.size:
        raise ValueError(f"Unexpected Oxford-IIIT Pet trimap labels: {unexpected.tolist()}")
    return (labels == 1).astype(np.float32), (labels != 3).astype(np.float32)


def preprocess_pair(
    image: Image.Image,
    trimap: Image.Image,
    image_size: int,
    *,
    augment: bool = False,
    flip: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize an image and trimap, then optionally flip both together.

    Images use bilinear interpolation and [0, 1] scaling. Trimaps use nearest
    interpolation so no new class IDs are introduced. A returned mask has
    shape ``(1, image_size, image_size)``.
    """

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if flip is None:
        flip = bool(torch.rand(()).item() < 0.5) if augment else False
    if flip and not augment:
        raise ValueError("flip=True requires augment=True")

    image = image.convert("RGB").resize(
        (image_size, image_size), resample=Image.Resampling.BILINEAR
    )
    trimap = trimap.resize(
        (image_size, image_size), resample=Image.Resampling.NEAREST
    )
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        trimap = trimap.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
    binary, valid = trimap_to_binary(trimap)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    binary_tensor = torch.from_numpy(binary.copy()).unsqueeze(0)
    valid_tensor = torch.from_numpy(valid.copy()).unsqueeze(0)
    return image_tensor, binary_tensor, valid_tensor
