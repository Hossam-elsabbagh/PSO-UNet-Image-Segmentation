"""Load the single project configuration file and resolve portable paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration, independent of the current working directory."""
    config_path = Path(path) if path is not None else PROJECT_ROOT / "config.json"
    with config_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def project_path(path: str | Path) -> Path:
    """Resolve a configured relative path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
