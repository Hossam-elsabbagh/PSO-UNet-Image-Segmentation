"""Per-image segmentation losses that exclude unclassified trimap pixels."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _check_inputs(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have identical shapes")
    if logits.ndim != 4:
        raise ValueError("expected tensors of shape (batch, channels, height, width)")
    valid = valid_mask > 0
    safe_targets = torch.where(valid, targets.to(logits.dtype), torch.zeros_like(logits))
    return valid, safe_targets


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Average BCE over valid pixels per image, then over valid images."""
    valid, safe_targets = _check_inputs(logits, targets, valid_mask)
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    pixel_losses = F.binary_cross_entropy_with_logits(
        safe_logits, safe_targets, reduction="none"
    )
    pixel_counts = valid.flatten(1).sum(dim=1)
    image_losses = (pixel_losses * valid).flatten(1).sum(dim=1) / pixel_counts.clamp_min(1)
    image_is_valid = pixel_counts > 0
    if not bool(image_is_valid.any()):
        return logits.sum() * 0.0
    return image_losses[image_is_valid].mean()


def masked_soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Average soft Dice loss over images with at least one valid pixel."""
    if smooth <= 0:
        raise ValueError("smooth must be positive")
    valid, safe_targets = _check_inputs(logits, targets, valid_mask)
    probabilities = torch.where(valid, torch.sigmoid(logits), torch.zeros_like(logits))
    intersection = (probabilities * safe_targets).flatten(1).sum(dim=1)
    denominator = probabilities.flatten(1).sum(dim=1) + safe_targets.flatten(1).sum(dim=1)
    image_losses = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    image_is_valid = valid.flatten(1).any(dim=1)
    if not bool(image_is_valid.any()):
        return logits.sum() * 0.0
    return image_losses[image_is_valid].mean()


def combined_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Weighted masked BCE and masked soft Dice with identical masking rules."""
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("loss weights must be nonnegative with a positive total")
    return (
        bce_weight * masked_bce_with_logits(logits, targets, valid_mask)
        + dice_weight * masked_soft_dice_loss(logits, targets, valid_mask)
    )
