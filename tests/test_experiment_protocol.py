"""Keep a completed test evaluation tied to its exact training runs."""

import json

import pytest

from scripts import evaluate_models
from src.utils import finish_experiment, sha256_file, start_experiment


def test_rerun_archives_stale_metadata_and_blocks_test_while_active(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"baseline": {"old": True}}), encoding="utf-8")
    run_id = start_experiment(path, "baseline")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert "baseline" not in metadata
    assert metadata["prior_runs"][0]["baseline"] == {"old": True}
    assert metadata["active_run"]["run_id"] == run_id

    finish_experiment(path, "baseline", run_id, {"summary": {"completed_epochs": 8}})
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert "active_run" not in metadata
    assert metadata["baseline"]["run_id"] == run_id


def test_completed_test_record_prevents_new_training(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"test": {"dice": 0.5}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already evaluated"):
        start_experiment(path, "pso")


@pytest.mark.parametrize("fault", ["active", "repeat", "config", "checkpoint"])
def test_test_split_stays_closed_when_protocol_is_invalid(tmp_path, monkeypatch, fault):
    metadata_path = tmp_path / "results" / "experiment_metadata.json"
    metadata_path.parent.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("baseline", "pso"):
        (checkpoint_dir / f"{name}_best.pt").write_bytes(name.encode())
    config = {"project": {"seed": 42}, "training": {"device": "cpu"}}
    recorded = config if fault != "config" else {"project": {"seed": 99}}
    metadata = {
        "baseline": {"config": recorded, "checkpoint_sha256": sha256_file(checkpoint_dir / "baseline_best.pt")},
        "pso_search": {"run_id": "pso-run"},
        "pso_final": {"config": recorded, "run_id": "pso-run",
                      "checkpoint_sha256": sha256_file(checkpoint_dir / "pso_best.pt")},
    }
    if fault == "active":
        metadata["active_run"] = {"section": "pso"}
    elif fault == "repeat":
        metadata["test"] = {"models": {}}
    elif fault == "checkpoint":
        metadata["baseline"]["checkpoint_sha256"] = "wrong"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(evaluate_models, "RESULTS_DIR", metadata_path.parent)
    monkeypatch.setattr(evaluate_models, "CHECKPOINTS_DIR", checkpoint_dir)
    monkeypatch.setattr(evaluate_models, "initialize", lambda _: (config, 42, "cpu"))
    monkeypatch.setattr(evaluate_models, "create_dataloaders",
                        lambda *_args, **_kwargs: pytest.fail("opened held-out test data"))
    monkeypatch.setattr("sys.argv", ["evaluate_models"])
    with pytest.raises(RuntimeError):
        evaluate_models.main()
