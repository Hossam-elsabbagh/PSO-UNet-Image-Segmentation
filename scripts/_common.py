"""Shared command-line setup for full and explicitly synthetic smoke runs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config, project_path
from src.unet import CompactUNet
from src.utils import select_device, set_seed


def add_common_arguments(parser: argparse.ArgumentParser, *, smoke: bool = False) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Alternative JSON configuration")
    parser.add_argument("--no-download", action="store_true", help="Require an existing local dataset")
    if smoke:
        parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic integration check only")


def initialize(config_path: Path | None):
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    set_seed(seed)
    device = select_device(config["training"]["device"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return config, seed, device


def build_model(config: dict, *, dropout: float | None = None) -> CompactUNet:
    settings = dict(config["model"])
    settings.pop("name", None)
    if dropout is not None:
        settings["dropout"] = dropout
    return CompactUNet(**settings)


def synthetic_loaders(seed: int) -> tuple[DataLoader, DataLoader]:
    """A labeled synthetic check, never a pet-dataset experiment."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand((4, 3, 16, 16), generator=generator)
    masks = (images[:, :1] > 0.5).float()
    valid = torch.ones_like(masks)
    train = DataLoader(TensorDataset(images[:2], masks[:2], valid[:2]), batch_size=1)
    validation = DataLoader(TensorDataset(images[2:], masks[2:], valid[2:]), batch_size=1)
    return train, validation


RESULTS_DIR = project_path("results")
CHECKPOINTS_DIR = project_path("checkpoints")
FIGURES_DIR = project_path("figures")
