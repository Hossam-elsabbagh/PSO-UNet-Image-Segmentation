"""Run the configured PSO pilot and train its selected final U-Net."""

from __future__ import annotations

import argparse
import json

from src.dataset import create_dataloaders
from src.evaluate import load_checkpoint_model
from src.pso import ParticleSwarmOptimizer, bounds_from_config, write_pso_history
from src.train import fit_model
from src.utils import (
    finish_experiment, runtime_context, set_seed, sha256_file, start_experiment,
    update_metadata, write_json,
)
from src.visualization import plot_predictions, plot_pso_convergence, plot_training_history

from ._common import (
    CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR, add_common_arguments,
    build_model, initialize, synthetic_loaders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, smoke=True)
    args = parser.parse_args()
    config, seed, device = initialize(args.config)
    pso_settings, training = config["pso"], config["training"]
    if not pso_settings["enabled"] and not args.smoke:
        raise ValueError("PSO is disabled in the configuration")
    if args.smoke:
        particles, rounds, candidate_epochs = 1, 1, 1
    else:
        particles = int(pso_settings["particles"])
        rounds = int(pso_settings["evaluation_rounds"])
        candidate_epochs = int(pso_settings["candidate_epochs"])
        # Confirm dataset access and fixed split records before starting search.
        create_dataloaders(config, download=not args.no_download, include_test=False)
        run_id = start_experiment(RESULTS_DIR / "experiment_metadata.json", "pso")

    optimizer = ParticleSwarmOptimizer(
        bounds_from_config(config), particles=particles, evaluation_rounds=rounds,
        inertia=float(pso_settings["inertia"]), cognitive=float(pso_settings["cognitive"]),
        social=float(pso_settings["social"]), seed=seed,
    )

    def objective(parameters: dict[str, float], evaluation_index: int) -> float:
        # Every candidate starts with fresh weights, fresh loaders, and the same seed.
        set_seed(seed)
        if args.smoke:
            train_loader, validation_loader = synthetic_loaders(seed)
        else:
            train_loader, validation_loader, _ = create_dataloaders(
                config, download=False, include_test=False
            )
        summary = fit_model(
            build_model(config, dropout=parameters["dropout"]),
            train_loader, validation_loader, epochs=candidate_epochs,
            learning_rate=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"], device=device,
            threshold=float(training["prediction_threshold"]),
            bce_weight=float(training["bce_weight"]),
            dice_weight=float(training["dice_weight"]),
        )
        return float(summary["best_validation_dice"])

    search = optimizer.optimize(objective)
    if args.smoke:
        print("SYNTHETIC PSO INTEGRATION CHECK ONLY: " + json.dumps({
            "evaluations": search["evaluations"],
            "best_candidate_validation_dice": search["best_candidate_validation_dice"],
        }))
        return

    write_pso_history(RESULTS_DIR / "pso_history.csv", search["history"])
    plot_pso_convergence(search["history"], FIGURES_DIR / "training_curves" / "pso_convergence.png")
    write_json(RESULTS_DIR / "best_pso_parameters.json", search["best_parameters"])
    update_metadata(RESULTS_DIR / "experiment_metadata.json", "pso_search", {
        **runtime_context(device, seed),
        "run_id": run_id,
        "config": config,
        "candidate_epochs": candidate_epochs,
        "evaluations": search["evaluations"],
        "search_seconds": search["search_seconds"],
        "best_candidate_validation_dice": search["best_candidate_validation_dice"],
        "best_parameters": search["best_parameters"],
    })

    # Match the full baseline budget after selecting hyperparameters.
    set_seed(seed)
    train_loader, validation_loader, _ = create_dataloaders(config, download=False, include_test=False)
    selected = search["best_parameters"]
    checkpoint = CHECKPOINTS_DIR / "pso_best.pt"
    summary = fit_model(
        build_model(config, dropout=selected["dropout"]),
        train_loader, validation_loader, epochs=int(training["epochs"]),
        learning_rate=selected["learning_rate"], weight_decay=selected["weight_decay"],
        device=device, checkpoint_path=checkpoint,
        history_path=RESULTS_DIR / "pso_training_history.csv",
        threshold=float(training["prediction_threshold"]),
        bce_weight=float(training["bce_weight"]), dice_weight=float(training["dice_weight"]),
    )
    plot_training_history(summary["history"], FIGURES_DIR / "training_curves", "pso_final")
    best_model, _ = load_checkpoint_model(checkpoint, device)
    plot_predictions(best_model, validation_loader, device,
                     FIGURES_DIR / "predictions" / "pso_validation.png",
                     threshold=float(training["prediction_threshold"]))
    finish_experiment(RESULTS_DIR / "experiment_metadata.json", "pso_final", run_id, {
        **runtime_context(device, seed),
        "config": config,
        "selected_hyperparameters": selected,
        "split": "validation",
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary": {key: value for key, value in summary.items() if key != "history"},
    })
    print(json.dumps({
        "selected_hyperparameters": selected,
        "search_seconds": search["search_seconds"],
        "final_training": {key: value for key, value in summary.items() if key != "history"},
    }, indent=2))


if __name__ == "__main__":
    main()
