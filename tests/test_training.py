"""Synthetic CPU integration checks for the shared training loop."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_checkpoint, load_checkpoint_model
from src.train import fit_model, run_epoch
from src.unet import CompactUNet


def _tiny_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(42)
    images = torch.rand(4, 3, 32, 32, generator=generator)
    targets = (images[:, :1] > 0.5).float()
    valid = torch.ones_like(targets)
    valid[:, :, :2, :] = 0
    return DataLoader(TensorDataset(images, targets, valid), batch_size=2)


def test_complete_training_loop_and_checkpoint(tmp_path) -> None:
    torch.manual_seed(42)
    model = CompactUNet(encoder_channels=(2, 4, 8, 16), bottleneck_channels=32)
    initial = model.output_conv.weight.detach().clone()
    loader = _tiny_loader()
    checkpoint_path = tmp_path / "model.pt"
    history_path = tmp_path / "history.csv"
    summary = fit_model(
        model,
        loader,
        loader,
        epochs=2,
        learning_rate=0.001,
        weight_decay=0.0001,
        device="cpu",
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )
    assert summary["completed_epochs"] == 2
    assert summary["best_epoch"] in (1, 2)
    assert 0 <= summary["best_validation_dice"] <= 1
    assert checkpoint_path.exists()
    assert history_path.exists()
    assert not torch.equal(model.output_conv.weight.detach(), initial)
    restored, checkpoint = load_checkpoint_model(checkpoint_path)
    assert checkpoint["epoch"] == summary["best_epoch"]
    scores = evaluate_checkpoint(checkpoint_path, loader, "cpu")
    assert scores["valid_images"] == 4
    assert 0 <= scores["dice"] <= 1
    assert scores["dice"] == pytest.approx(summary["best_validation_dice"])
    assert restored.get_config()["encoder_channels"] == (2, 4, 8, 16)


def test_empty_valid_batch_skips_optimizer_update() -> None:
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    images = torch.rand(2, 3, 8, 8)
    targets = torch.zeros(2, 1, 8, 8)
    valid = torch.zeros_like(targets)
    loader = DataLoader(TensorDataset(images, targets, valid), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = run_epoch(model, loader, "cpu", optimizer=optimizer)
    assert result["skipped_batches"] == 1
    assert result["valid_images"] == 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
