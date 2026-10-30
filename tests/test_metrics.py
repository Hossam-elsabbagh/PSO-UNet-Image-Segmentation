"""Per-image foreground Dice and IoU policy checks."""

import pytest
import torch

from src.metrics import segmentation_metrics


def _score(prediction, target, valid=None):
    prediction = torch.tensor(prediction, dtype=torch.float32).reshape(1, 1, 2, 2)
    target = torch.tensor(target, dtype=torch.float32).reshape(1, 1, 2, 2)
    valid = (
        torch.ones_like(target)
        if valid is None
        else torch.tensor(valid, dtype=torch.float32).reshape(1, 1, 2, 2)
    )
    return segmentation_metrics(prediction, target, valid, from_logits=False)


def test_perfect_segmentation() -> None:
    scores = _score([1, 0, 0, 1], [1, 0, 0, 1])
    assert scores["dice"] == 1.0
    assert scores["iou"] == 1.0


def test_partial_overlap() -> None:
    scores = _score([1, 1, 0, 0], [1, 0, 1, 0])
    assert scores["dice"] == pytest.approx(0.5)
    assert scores["iou"] == pytest.approx(1 / 3)


def test_no_overlap() -> None:
    scores = _score([1, 0, 0, 0], [0, 1, 0, 0])
    assert scores["dice"] == 0.0
    assert scores["iou"] == 0.0


def test_ignored_pixels_are_excluded() -> None:
    scores = _score([1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 1, 1])
    assert scores["dice"] == 1.0
    assert scores["iou"] == 1.0


def test_both_empty_foregrounds_score_one() -> None:
    scores = _score([0, 0, 0, 0], [0, 0, 0, 0])
    assert scores["dice"] == 1.0
    assert scores["iou"] == 1.0


def test_zero_valid_image_is_excluded_from_mean() -> None:
    prediction = torch.tensor([[[[1.0]]], [[[0.0]]]])
    target = torch.tensor([[[[1.0]]], [[[1.0]]]])
    valid = torch.tensor([[[[1.0]]], [[[0.0]]]])
    scores = segmentation_metrics(prediction, target, valid, from_logits=False)
    assert scores == {"dice": 1.0, "iou": 1.0, "valid_images": 1}
