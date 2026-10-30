"""Shared baseline and PSO-candidate training loop."""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import combined_segmentation_loss
from .metrics import per_image_scores

LOGGER = logging.getLogger(__name__)


def _write_history(path: str | Path, rows: list[dict[str, float | int]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> dict[str, float | int]:
    """Run one complete epoch and aggregate metrics equally per valid image.

    A batch with no valid pixels is skipped, including its optimizer update.
    The returned loss is the mean of per-image masked losses, so batch size does
    not change the weighting of the reported epoch loss.
    """
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    valid_images = 0
    skipped_batches = 0
    batch_count = 0
    device = torch.device(device)

    for images, targets, valid_mask in dataloader:
        batch_count += 1
        images = images.to(device)
        targets = targets.to(device)
        valid_mask = valid_mask.to(device)
        batch_valid = int((valid_mask.flatten(1) > 0).any(dim=1).sum().item())
        if batch_valid == 0:
            skipped_batches += 1
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = combined_segmentation_loss(
                logits,
                targets,
                valid_mask,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )
            if training:
                loss.backward()
                optimizer.step()

        dice, iou, included = per_image_scores(
            logits.detach(), targets, valid_mask, threshold=threshold
        )
        loss_sum += float(loss.detach().item()) * batch_valid
        dice_sum += float(dice[included].sum().item())
        iou_sum += float(iou[included].sum().item())
        valid_images += batch_valid

    return {
        "loss": loss_sum / valid_images if valid_images else float("nan"),
        "dice": dice_sum / valid_images if valid_images else float("nan"),
        "iou": iou_sum / valid_images if valid_images else float("nan"),
        "valid_images": valid_images,
        "skipped_batches": skipped_batches,
        "batches": batch_count,
    }


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    threshold: float = 0.5,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> dict[str, Any]:
    """Fit a fresh model with AdamW and select the best validation Dice.

    The caller controls model initialization and random seeds. This function is
    used unchanged for the baseline, PSO candidates, and final PSO model.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")

    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_dice = -math.inf
    best_epoch = 0
    best_validation: dict[str, float | int] | None = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            threshold=threshold,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )
        if train_metrics["valid_images"] == 0:
            raise ValueError("training split contains no images with valid pixels")

        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            threshold=threshold,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )
        if validation_metrics["valid_images"] == 0:
            raise ValueError("validation split contains no images with valid pixels")

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_dice": float(train_metrics["dice"]),
            "train_iou": float(train_metrics["iou"]),
            "val_loss": float(validation_metrics["loss"]),
            "val_dice": float(validation_metrics["dice"]),
            "val_iou": float(validation_metrics["iou"]),
            "train_valid_images": int(train_metrics["valid_images"]),
            "val_valid_images": int(validation_metrics["valid_images"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("train_loss", "val_loss", "val_dice", "val_iou")
        ):
            raise RuntimeError(f"non-finite training or validation result at epoch {epoch}")
        history.append(row)
        if history_path is not None:
            _write_history(history_path, history)
        LOGGER.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_dice=%.4f val_iou=%.4f",
            epoch,
            epochs,
            row["train_loss"],
            row["val_loss"],
            row["val_dice"],
            row["val_iou"],
        )

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            best_epoch = epoch
            best_validation = dict(validation_metrics)
            if checkpoint_path is not None:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": (
                            model.get_config() if hasattr(model, "get_config") else None
                        ),
                        "epoch": epoch,
                        "validation_metrics": best_validation,
                        "training_settings": {
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "threshold": threshold,
                            "bce_weight": bce_weight,
                            "dice_weight": dice_weight,
                        },
                    },
                    temporary,
                )
                temporary.replace(path)

    return {
        "history": history,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "best_validation_iou": (
            float(best_validation["iou"]) if best_validation else float("nan")
        ),
        "best_validation_loss": (
            float(best_validation["loss"]) if best_validation else float("nan")
        ),
        "training_seconds": time.perf_counter() - started,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
