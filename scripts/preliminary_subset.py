"""Run a clearly labeled, real-data Phase 1 pipeline check on small fixed subsets.

This is not the eight-epoch baseline or PSO comparison. It gives an initial
measured Oxford-IIIT Pet result when a full CPU experiment is impractical.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import create_dataloaders
from src.evaluate import load_checkpoint_model
from src.train import fit_model
from src.utils import runtime_context, set_seed, sha256_file, update_metadata
from src.visualization import plot_predictions, plot_training_history

from ._common import (
    CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR, add_common_arguments,
    build_model, initialize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    config, seed, device = initialize(args.config)
    if args.train_samples < 1 or args.validation_samples < 1 or args.epochs < 1:
        raise ValueError("Sample counts and epochs must be positive")
    metadata_path = RESULTS_DIR / "experiment_metadata.json"
    if metadata_path.exists() and "test" in json.loads(metadata_path.read_text(encoding="utf-8")):
        raise RuntimeError("Held-out test was already evaluated; do not run new model selection")
    full_train, full_validation, _ = create_dataloaders(
        config, download=not args.no_download, include_test=False
    )
    if args.train_samples > len(full_train.dataset) or args.validation_samples > len(full_validation.dataset):
        raise ValueError("Requested preliminary subset exceeds the fixed split")
    batch_size = int(config["dataset"]["batch_size"])
    train_positions = torch.randperm(
        len(full_train.dataset), generator=torch.Generator().manual_seed(seed)
    )[:args.train_samples].tolist()
    validation_positions = torch.randperm(
        len(full_validation.dataset), generator=torch.Generator().manual_seed(seed + 1)
    )[:args.validation_samples].tolist()
    train_loader = DataLoader(
        Subset(full_train.dataset, train_positions), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        Subset(full_validation.dataset, validation_positions), batch_size=batch_size,
        shuffle=False,
    )
    set_seed(seed)
    settings = config["training"]
    checkpoint = CHECKPOINTS_DIR / "preliminary_best.pt"
    summary = fit_model(
        build_model(config), train_loader, validation_loader, epochs=args.epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]), device=device,
        checkpoint_path=checkpoint, history_path=RESULTS_DIR / "preliminary_history.csv",
        threshold=float(settings["prediction_threshold"]),
        bce_weight=float(settings["bce_weight"]), dice_weight=float(settings["dice_weight"]),
    )
    plot_training_history(summary["history"], FIGURES_DIR / "training_curves", "preliminary_subset")
    # The selected checkpoint is loaded for the measured prediction figure.
    best_model, _ = load_checkpoint_model(checkpoint, device)
    plot_predictions(best_model, validation_loader, device,
                     FIGURES_DIR / "predictions" / "preliminary_validation.png",
                     count=2,
                     threshold=float(settings["prediction_threshold"]))
    train_ids = (RESULTS_DIR.parent / "splits" / "train.txt").read_text(encoding="utf-8").splitlines()
    validation_ids = (RESULTS_DIR.parent / "splits" / "validation.txt").read_text(encoding="utf-8").splitlines()
    update_metadata(metadata_path, "preliminary_subset", {
        **runtime_context(device, seed),
        "label": "Non-comparable preliminary Oxford subset; not the full baseline",
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "train_ids": [train_ids[index] for index in train_positions],
        "validation_ids": [validation_ids[index] for index in validation_positions],
        "config": config,
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary": {key: value for key, value in summary.items() if key != "history"},
    })
    print("PRELIMINARY OXFORD SUBSET ONLY: " + json.dumps({
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "epochs": args.epochs,
        "validation_dice": summary["best_validation_dice"],
        "validation_iou": summary["best_validation_iou"],
        "training_seconds": summary["training_seconds"],
    }))


if __name__ == "__main__":
    main()
