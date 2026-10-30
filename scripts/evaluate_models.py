"""Evaluate both finalized checkpoints once on the held-out official test set."""

from __future__ import annotations

import argparse
import json

from src.dataset import create_dataloaders
from src.evaluate import evaluate_checkpoint, load_checkpoint_model
from src.utils import runtime_context, sha256_file, update_metadata
from src.visualization import plot_predictions

from ._common import (
    CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR, add_common_arguments, initialize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    config, seed, device = initialize(args.config)
    metadata_path = RESULTS_DIR / "experiment_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Train the baseline and final PSO-selected model before opening the test split")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "active_run" in metadata:
        raise RuntimeError("A model experiment is still in progress; final test evaluation is locked")
    if "test" in metadata:
        raise RuntimeError("Held-out test evaluation was already recorded; do not repeat it")
    if "baseline" not in metadata or "pso_final" not in metadata:
        raise RuntimeError("Both final models must finish validation selection before test evaluation")
    if metadata["baseline"]["config"] != metadata["pso_final"]["config"]:
        raise RuntimeError("Final models used different project configurations; review comparability")
    if config != metadata["baseline"]["config"]:
        raise RuntimeError("Current config differs from the completed training protocol")
    if metadata.get("pso_search", {}).get("run_id") != metadata["pso_final"].get("run_id"):
        raise RuntimeError("PSO search and final model belong to different runs")
    for name in ("baseline", "pso"):
        checkpoint = CHECKPOINTS_DIR / f"{name}_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing final {name} checkpoint")
        section = "baseline" if name == "baseline" else "pso_final"
        if sha256_file(checkpoint) != metadata[section]["checkpoint_sha256"]:
            raise RuntimeError(f"{name} checkpoint differs from its completed metadata")

    _, _, test_loader = create_dataloaders(
        config, download=not args.no_download, include_test=True
    )
    assert test_loader is not None
    settings = config["training"]
    outcomes = {}
    for name in ("baseline", "pso"):
        checkpoint = CHECKPOINTS_DIR / f"{name}_best.pt"
        metrics = evaluate_checkpoint(
            checkpoint, test_loader, device,
            threshold=float(settings["prediction_threshold"]),
            bce_weight=float(settings["bce_weight"]),
            dice_weight=float(settings["dice_weight"]),
        )
        outcomes[name] = metrics
        model, _ = load_checkpoint_model(checkpoint, device)
        plot_predictions(model, test_loader, device,
                         FIGURES_DIR / "predictions" / f"{name}_test.png",
                         threshold=float(settings["prediction_threshold"]))
    update_metadata(metadata_path, "test", {
        **runtime_context(device, seed),
        "split": "official held-out test",
        "models": outcomes,
    })
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()
