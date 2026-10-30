"""Export a comparison table from recorded experiments; never fill in missing runs."""

from __future__ import annotations

import csv
import json

from src.visualization import plot_model_comparison

from ._common import FIGURES_DIR, RESULTS_DIR


def collect_rows(metadata: dict) -> list[dict]:
    """Build table rows from measured final-model metadata only."""
    rows = []
    for key, model in (("baseline", "Baseline U-Net"), ("pso_final", "PSO-Selected U-Net")):
        if key not in metadata:
            continue
        summary = metadata[key]["summary"]
        rows.append({
            "model": model,
            "split": "validation",
            "epochs": summary["completed_epochs"],
            "dice": summary["best_validation_dice"],
            "iou": summary["best_validation_iou"],
            "training_seconds": summary["training_seconds"],
            "search_seconds": metadata.get("pso_search", {}).get("search_seconds") if key == "pso_final" else "",
        })
    if "test" in metadata:
        for key, model in (("baseline", "Baseline U-Net"), ("pso", "PSO-Selected U-Net")):
            if key not in metadata["test"]["models"]:
                continue
            final = metadata["baseline"] if key == "baseline" else metadata["pso_final"]
            metrics = metadata["test"]["models"][key]
            rows.append({
                "model": model,
                "split": "test",
                "epochs": final["summary"]["completed_epochs"],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "training_seconds": final["summary"]["training_seconds"],
                "search_seconds": metadata.get("pso_search", {}).get("search_seconds") if key == "pso" else "",
            })
    return rows


def main() -> None:
    metadata_path = RESULTS_DIR / "experiment_metadata.json"
    if not metadata_path.exists():
        print("No completed experiment metadata; results.csv was not created.")
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = collect_rows(metadata)
    if not rows:
        print("No completed final-model results; results.csv was not created.")
        return
    output = RESULTS_DIR / "results.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in ("validation", "test"):
        comparable = [row for row in rows if row["split"] == split]
        if len(comparable) == 2:
            plot_model_comparison(comparable,
                                  FIGURES_DIR / "training_curves" / f"comparison_{split}.png")
    print(f"Wrote {len(rows)} measured rows to {output}")


if __name__ == "__main__":
    main()
