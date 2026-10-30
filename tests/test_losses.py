"""Masked segmentation loss checks."""

import pytest
import torch

from src.losses import (
    combined_segmentation_loss,
    masked_bce_with_logits,
    masked_soft_dice_loss,
)


@pytest.mark.parametrize(
    "loss_fn", [masked_bce_with_logits, masked_soft_dice_loss, combined_segmentation_loss]
)
def test_ignored_pixels_do_not_change_loss(loss_fn) -> None:
    logits = torch.tensor([[[[2.0, -1.0], [0.5, -3.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [3.0, 3.0]]]])
    valid = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    changed_logits = logits.clone()
    changed_logits[0, 0, 1] = torch.tensor([9.0, -9.0])
    changed_targets = targets.clone()
    changed_targets[0, 0, 1] = torch.tensor([0.0, 1.0])
    assert torch.allclose(
        loss_fn(logits, targets, valid),
        loss_fn(changed_logits, changed_targets, valid),
    )


def test_bce_uses_valid_pixel_mean_per_image() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    assert masked_bce_with_logits(logits, targets, valid).item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )


@pytest.mark.parametrize(
    "loss_fn", [masked_bce_with_logits, masked_soft_dice_loss, combined_segmentation_loss]
)
def test_zero_valid_image_is_excluded_from_batch_mean(loss_fn) -> None:
    logits = torch.tensor([[[[0.3, -0.2]]], [[[10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    valid = torch.tensor([[[[1.0, 1.0]]], [[[0.0, 0.0]]]])
    expected = loss_fn(logits[:1], targets[:1], valid[:1])
    assert torch.allclose(loss_fn(logits, targets, valid), expected)


def test_all_ignored_batch_has_zero_loss_and_zero_gradient() -> None:
    logits = torch.randn(2, 1, 2, 2, requires_grad=True)
    targets = torch.full_like(logits, 3.0)
    valid = torch.zeros_like(logits)
    loss = combined_segmentation_loss(logits, targets, valid)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
