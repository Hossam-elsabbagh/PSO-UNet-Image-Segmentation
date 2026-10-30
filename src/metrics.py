"""Foreground Dice and IoU using valid pixels and per-image aggregation."""

from __future__ import annotations

import torch


def per_image_scores(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Dice, IoU, and an inclusion mask for each image in a batch.

    Both empty foreground sets score 1. Images with no valid pixels are marked
    excluded and must not contribute to a dataset mean.
    """
    if predictions.shape != targets.shape or predictions.shape != valid_mask.shape:
        raise ValueError("predictions, targets, and valid_mask must have identical shapes")
    if predictions.ndim != 4:
        raise ValueError("expected tensors of shape (batch, channels, height, width)")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    with torch.no_grad():
        probabilities = predictions.sigmoid() if from_logits else predictions
        valid = valid_mask > 0
        predicted = (probabilities >= threshold) & valid
        actual = (targets > 0.5) & valid
        tp = (predicted & actual).flatten(1).sum(dim=1).to(torch.float64)
        pred_count = predicted.flatten(1).sum(dim=1).to(torch.float64)
        actual_count = actual.flatten(1).sum(dim=1).to(torch.float64)
        dice_denominator = pred_count + actual_count
        iou_denominator = pred_count + actual_count - tp
        dice = torch.where(
            dice_denominator > 0,
            2.0 * tp / dice_denominator.clamp_min(1),
            torch.ones_like(tp),
        )
        iou = torch.where(
            iou_denominator > 0,
            tp / iou_denominator.clamp_min(1),
            torch.ones_like(tp),
        )
        included = valid.flatten(1).any(dim=1)
        return dice, iou, included


def segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, float | int]:
    """Mean per-image Dice and IoU, excluding zero-valid-pixel images."""
    dice, iou, included = per_image_scores(
        predictions, targets, valid_mask, threshold=threshold, from_logits=from_logits
    )
    count = int(included.sum().item())
    return {
        "dice": float(dice[included].mean().item()) if count else float("nan"),
        "iou": float(iou[included].mean().item()) if count else float("nan"),
        "valid_images": count,
    }
