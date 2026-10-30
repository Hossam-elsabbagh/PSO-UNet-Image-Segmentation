"""Figures generated only from completed model evaluations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history: list[dict], output_dir: str | Path, prefix: str) -> list[Path]:
    """Plot actual training/validation loss and validation Dice by epoch."""
    if not history:
        raise ValueError("Cannot plot an empty training history")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    destinations = [output / f"{prefix}_loss.png", output / f"{prefix}_validation_dice.png"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Training loss", marker="o")
    ax.plot(epochs, [row["val_loss"] for row in history], label="Validation loss", marker="o")
    ax.set(xlabel="Epoch", ylabel="Masked BCE + Dice loss", title=f"{prefix.title()} loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destinations[0], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [row["val_dice"] for row in history], marker="o", color="#2058a5")
    ax.set(xlabel="Epoch", ylabel="Mean foreground Dice", ylim=(0, 1),
           title=f"{prefix.title()} validation Dice")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destinations[1], dpi=160)
    plt.close(fig)
    return destinations


def plot_pso_convergence(history: list[dict], output_path: str | Path) -> Path:
    """Plot measured global-best Dice after each candidate evaluation."""
    if not history:
        raise ValueError("Cannot plot empty PSO history")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step([row["evaluation"] for row in history],
            [row["global_best_dice"] for row in history], where="post", color="#2058a5")
    ax.scatter([row["evaluation"] for row in history],
               [row["validation_dice"] for row in history], color="#dc784b", s=24,
               label="Candidate Dice")
    ax.set(xlabel="Candidate evaluation", ylabel="Validation Dice", ylim=(0, 1),
           title="PSO pilot convergence")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def plot_model_comparison(rows: list[dict], output_path: str | Path) -> Path:
    """Compare measured final-model Dice and IoU on the same split."""
    if len(rows) != 2 or rows[0]["split"] != rows[1]["split"]:
        raise ValueError("Comparison needs two final models evaluated on the same split")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [float(row["dice"]) for row in rows], width, label="Dice")
    ax.bar(x + width / 2, [float(row["iou"]) for row in rows], width, label="Foreground IoU")
    ax.set(xticks=x, xticklabels=[row["model"] for row in rows], ylabel="Mean score",
           ylim=(0, 1), title=f"Final models on {rows[0]['split']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


@torch.no_grad()
def plot_predictions(
    model: torch.nn.Module,
    dataloader,
    device: torch.device | str,
    output_path: str | Path,
    *,
    count: int = 3,
    threshold: float = 0.5,
) -> Path:
    """Show actual model predictions beside validation images and masks."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    images, masks, valid = next(iter(dataloader))
    count = min(count, len(images))
    if count < 1:
        raise ValueError("Prediction loader is empty")
    model.eval()
    predicted = (torch.sigmoid(model(images[:count].to(device))) >= threshold).cpu()
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    for index in range(count):
        image = images[index].permute(1, 2, 0).cpu().numpy()
        truth = np.ma.masked_where(valid[index, 0].cpu().numpy() == 0,
                                   masks[index, 0].cpu().numpy())
        prediction = np.ma.masked_where(valid[index, 0].cpu().numpy() == 0,
                                        predicted[index, 0].numpy())
        for col, (array, title) in enumerate((
            (image, "Image"),
            (truth, "Ground truth"),
            (valid[index, 0].cpu().numpy(), "Valid pixels"),
            (prediction, "Prediction"),
        )):
            axes[index, col].imshow(array, cmap=None if col == 0 else "gray", vmin=0, vmax=1)
            axes[index, col].set_title(title)
            axes[index, col].axis("off")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target
