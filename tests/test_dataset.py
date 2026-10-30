"""Synthetic split and wrapper tests; no dataset download required."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import (
    OxfordPetBinaryDataset,
    _read_or_write_split,
    _sample_ids,
    make_train_validation_ids,
    prepare_data,
)


class FakePetSource:
    def __init__(self, root: Path) -> None:
        self._images = [root / f"pet_{i}.jpg" for i in range(2)]
        self._segs = [root / f"pet_{i}.png" for i in range(2)]
        for path in (*self._images, *self._segs):
            path.touch()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image]:
        image = Image.new("RGB", (3, 2), "red")
        trimap = Image.fromarray(np.array([[1, 2, 3], [2, 1, 3]], dtype=np.uint8))
        return image, trimap


def test_official_size_split_is_deterministic_and_disjoint() -> None:
    ids = [f"pet_{index}" for index in range(3680)]
    first = make_train_validation_ids(ids, validation_ratio=0.2, seed=42)
    second = make_train_validation_ids(ids, validation_ratio=0.2, seed=42)
    assert first == second
    train, validation = first
    assert len(train) == 2944
    assert len(validation) == 736
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(ids)
    assert set(ids).isdisjoint({f"test_{index}" for index in range(3669)})


def test_split_records_are_reused_and_guard_against_changed_seed(tmp_path: Path) -> None:
    path = tmp_path / "splits" / "train.txt"
    assert _read_or_write_split(path, ["a", "b"]) == ["a", "b"]
    assert _read_or_write_split(path, ["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="Persisted split"):
        _read_or_write_split(path, ["b", "a"])


def test_image_trimap_pairing_and_binary_wrapper(tmp_path: Path) -> None:
    source = FakePetSource(tmp_path)
    assert _sample_ids(source) == ["pet_0", "pet_1"]
    dataset = OxfordPetBinaryDataset(source, [1], 8)
    image, target, valid = dataset[0]
    assert len(dataset) == 1
    assert image.shape == (3, 8, 8)
    assert target.shape == valid.shape == (1, 8, 8)
    assert target.dtype == valid.dtype == torch.float32
    assert valid.min() == 0 and valid.max() == 1


def test_pair_validation_rejects_missing_trimap(tmp_path: Path) -> None:
    source = FakePetSource(tmp_path)
    source._segs[0].unlink()
    with pytest.raises(FileNotFoundError, match="Missing pet image or trimap"):
        _sample_ids(source)


def test_prepare_data_keeps_test_unopened_until_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MetadataSource:
        def __init__(self, split: str) -> None:
            self.ids = [f"{split}_{index}" for index in range(3680 if split == "trainval" else 3669)]

        def __len__(self) -> int:
            return len(self.ids)

    requested: list[str] = []

    def fake_load(_root: Path, split: str, _download: bool) -> MetadataSource:
        requested.append(split)
        return MetadataSource(split)

    monkeypatch.setattr("src.dataset._load_source", fake_load)
    monkeypatch.setattr("src.dataset._sample_ids", lambda source: source.ids)
    config = {
        "project": {"seed": 42},
        "dataset": {
            "root": "data/raw",
            "validation_ratio": 0.2,
            "image_size": 128,
            "batch_size": 16,
        },
    }
    development = prepare_data(config, project_root=tmp_path, download=False)
    assert requested == ["trainval"]
    assert not (tmp_path / "splits/test.txt").exists()
    assert len(development.train_indices) == 2944
    assert len(development.validation_indices) == 736

    complete = prepare_data(config, project_root=tmp_path, download=False, include_test=True)
    assert requested == ["trainval", "trainval", "test"]
    assert len(complete.test_ids) == 3669
    assert (tmp_path / "splits/test.txt").exists()
    assert set(complete.test_ids).isdisjoint(complete.train_ids)
    assert set(complete.test_ids).isdisjoint(complete.validation_ids)
