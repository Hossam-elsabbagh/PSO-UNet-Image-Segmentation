"""Synthetic checks for image, trimap, and augmentation semantics."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.preprocessing import preprocess_pair, trimap_to_binary


def test_trimap_mapping_preserves_ignore() -> None:
    trimap = np.array([[1, 2, 3], [3, 1, 2]], dtype=np.uint8)
    binary, valid = trimap_to_binary(trimap)
    np.testing.assert_array_equal(binary, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(valid, [[1, 1, 0], [0, 1, 1]])


def test_unknown_trimap_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        trimap_to_binary(np.array([[0, 1]], dtype=np.uint8))


def test_pair_shapes_scaling_and_nearest_neighbor() -> None:
    image = Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8))
    trimap = Image.fromarray(np.array([[1, 3], [2, 2]], dtype=np.uint8))
    image_t, target_t, valid_t = preprocess_pair(image, trimap, 4)
    assert image_t.shape == (3, 4, 4)
    assert target_t.shape == valid_t.shape == (1, 4, 4)
    assert image_t.dtype == target_t.dtype == valid_t.dtype == torch.float32
    assert torch.all(image_t == 1)
    assert set(torch.unique(target_t).tolist()) == {0, 1}
    assert set(torch.unique(valid_t).tolist()) == {0, 1}
    assert valid_t[0, 0, 3].item() == 0


def test_flip_is_synchronized_across_image_and_masks() -> None:
    pixels = np.zeros((2, 4, 3), dtype=np.uint8)
    pixels[:, 0] = [255, 0, 0]
    trimap = np.full((2, 4), 2, dtype=np.uint8)
    trimap[:, 0] = 1
    trimap[:, 1] = 3
    image = Image.fromarray(pixels)
    mask = Image.fromarray(trimap)
    normal = preprocess_pair(image, mask, 4, augment=True, flip=False)
    flipped = preprocess_pair(image, mask, 4, augment=True, flip=True)
    for original, mirrored in zip(normal, flipped):
        assert torch.equal(torch.flip(original, dims=(-1,)), mirrored)


def test_all_ignored_pixels_remain_invalid() -> None:
    image = Image.new("RGB", (2, 2), "black")
    trimap = Image.fromarray(np.full((2, 2), 3, dtype=np.uint8))
    _, target, valid = preprocess_pair(image, trimap, 4)
    assert target.sum() == 0
    assert valid.sum() == 0
