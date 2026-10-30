"""Small shared helpers for reproducible experiment execution."""

from __future__ import annotations

import json
import logging
import platform
import random
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without enabling slow global kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a requested execution device, with an explicit CPU fallback."""
    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; using CPU")
        preference = "cpu"
    return torch.device(preference)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a readable JSON artifact, creating its parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def update_metadata(path: str | Path, section: str, record: dict[str, Any]) -> None:
    """Upsert one completed run without discarding other experiment metadata."""
    target = Path(path)
    if target.exists():
        with target.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = {}
    payload[section] = record
    write_json(target, payload)


def start_experiment(path: str | Path, section: str) -> str:
    """Invalidate stale final records before a rerun and keep an audit copy."""
    if section not in {"baseline", "pso"}:
        raise ValueError("Unknown experiment section")
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    if "test" in payload:
        raise RuntimeError(
            "The held-out test was already evaluated. Preserve this completed protocol; "
            "start a separately documented experiment to rerun training."
        )
    stale_keys = ("baseline",) if section == "baseline" else ("pso_search", "pso_final")
    stale = {key: payload.pop(key) for key in stale_keys if key in payload}
    if "active_run" in payload:
        stale["active_run"] = payload.pop("active_run")
    if stale:
        payload.setdefault("prior_runs", []).append(stale)
    run_id = uuid.uuid4().hex
    payload["active_run"] = {
        "section": section,
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(target, payload)
    return run_id


def finish_experiment(path: str | Path, section: str, run_id: str, record: dict[str, Any]) -> None:
    """Mark a successful final run and remove its active-run guard."""
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    active = payload.get("active_run", {})
    expected = "baseline" if section == "baseline" else "pso"
    if active.get("section") != expected or active.get("run_id") != run_id:
        raise RuntimeError("Experiment run ID no longer matches active metadata")
    record["run_id"] = run_id
    payload[section] = record
    payload.pop("active_run")
    write_json(target, payload)


def sha256_file(path: str | Path) -> str:
    """Hash a selected checkpoint so metadata cannot silently describe new weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_context(device: torch.device, seed: int) -> dict[str, Any]:
    """Describe the actual environment used by an experiment."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "seed": seed,
    }
