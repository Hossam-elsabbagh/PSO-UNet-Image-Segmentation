"""Train the configured fixed-hyperparameter baseline on Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import json

from src.dataset import create_dataloaders
from src.evaluate import load_checkpoint_model
from src.train import fit_model
from src.utils import finish_experiment, runtime_context, sha256_file, start_experiment
from src.visualization import plot_predictions, plot_training_history

from ._common import (
    CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR, add_common_arguments,
    build_model, initialize, synthetic_loaders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, smoke=True)
    args = parser.parse_args()
    config, seed, device = initialize(args.config)
    settings = config["training"]

    if args.smoke:
        train_loader, validation_loader = synthetic_loaders(seed)
        summary = fit_model(
            build_model(config), train_loader, validation_loader, epochs=1,
            learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
            device=device, threshold=settings["prediction_threshold"],
            bce_weight=settings["bce_weight"], dice_weight=settings["dice_weight"],
        )
        print("SYNTHETIC SMOKE TEST ONLY: " + json.dumps({
            "epochs": summary["completed_epochs"],
            "validation_dice": summary["best_validation_dice"],
        }))
        return

    train_loader, validation_loader, _ = create_dataloaders(
        config, download=not args.no_download, include_test=False
    )
    metadata_path = RESULTS_DIR / "experiment_metadata.json"
    run_id = start_experiment(metadata_path, "baseline")
    checkpoint = CHECKPOINTS_DIR / "baseline_best.pt"
    summary = fit_model(
        build_model(config), train_loader, validation_loader,
        epochs=int(settings["epochs"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        device=device, checkpoint_path=checkpoint,
        history_path=RESULTS_DIR / "training_history.csv",
        threshold=float(settings["prediction_threshold"]),
        bce_weight=float(settings["bce_weight"]), dice_weight=float(settings["dice_weight"]),
    )
    plot_training_history(summary["history"], FIGURES_DIR / "training_curves", "baseline")
    selected, _ = load_checkpoint_model(checkpoint, device)
    plot_predictions(selected, validation_loader, device,
                     FIGURES_DIR / "predictions" / "baseline_validation.png",
                     threshold=float(settings["prediction_threshold"]))
    finish_experiment(metadata_path, "baseline", run_id, {
        **runtime_context(device, seed),
        "config": config,
        "dataset": "Oxford-IIIT Pet official trainval split",
        "split": "validation",
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary": {key: value for key, value in summary.items() if key != "history"},
    })
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
