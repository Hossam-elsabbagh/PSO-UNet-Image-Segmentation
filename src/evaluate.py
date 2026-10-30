"""Evaluate trained U-Net models on a validation or held-out test loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .train import run_epoch
from .unet import CompactUNet


def load_checkpoint_model(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[CompactUNet, dict]:
    """Recreate a model from a checkpoint written by :func:`fit_model`."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise ValueError("checkpoint does not contain a CompactUNet configuration")
    model = CompactUNet(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(
    model: CompactUNet,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    threshold: float = 0.5,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate the same masked loss and per-image metrics used in training."""
    model.to(device)
    return run_epoch(
        model,
        dataloader,
        device,
        threshold=threshold,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    threshold: float = 0.5,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> dict[str, float | int]:
    """Load one selected checkpoint and evaluate it without changing weights."""
    model, _ = load_checkpoint_model(checkpoint_path, device)
    return evaluate_model(
        model,
        dataloader,
        device,
        threshold=threshold,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
