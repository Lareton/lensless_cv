import sys
from types import SimpleNamespace

import numpy as np
import torch

from src.datasets.digicam import DigiCamDataset
from src.datasets.psf import PSFStore


class FakeDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def select(self, indices):
        return FakeDataset([self.items[index] for index in indices])


def test_digicam_dataset(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    data_dir.mkdir()
    mask_dir.mkdir()
    (data_dir / "test-00000-of-00001.parquet").touch()
    np.save(mask_dir / "mask_7.npy", np.ones((2, 2), dtype=np.float32))

    items = [
        {
            "lensless": np.full((300, 400, 3), 32, dtype=np.uint8),
            "lensed": np.full((20, 30, 3), 64, dtype=np.uint8),
            "mask_label": 7,
        }
        for _ in range(3)
    ]

    datasets_module = SimpleNamespace(
        load_dataset=lambda *args, **kwargs: FakeDataset(items)
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setattr(
        PSFStore,
        "_build",
        lambda self, mask_id: torch.ones(3, 300, 400),
    )

    dataset = DigiCamDataset(
        "test",
        data_dir=tmp_path,
        save_psf=False,
        limit=1,
    )
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image_id"] == "test_00000"
    assert item["mask_label"] == 7
    assert item["lensless"].shape == (3, 300, 400)
    assert item["target"].shape == (3, 300, 400)
    assert item["target_roi"].shape == (3, 200, 266)


def test_digicam_offset(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    data_dir.mkdir()
    mask_dir.mkdir()
    (data_dir / "train-00000-of-00001.parquet").touch()
    np.save(mask_dir / "mask_7.npy", np.ones((2, 2), dtype=np.float32))
    items = [
        {
            "lensless": np.full((300, 400, 3), index, dtype=np.uint8),
            "lensed": np.full((20, 30, 3), index, dtype=np.uint8),
            "mask_label": 7,
        }
        for index in range(5)
    ]
    datasets_module = SimpleNamespace(
        load_dataset=lambda *args, **kwargs: FakeDataset(items)
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setattr(
        PSFStore,
        "_build",
        lambda self, mask_id: torch.ones(3, 300, 400),
    )

    dataset = DigiCamDataset(
        "train",
        data_dir=tmp_path,
        save_psf=False,
        offset=3,
        limit=1,
    )

    assert len(dataset) == 1
    assert dataset[0]["image_id"] == "train_00003"


def test_digicam_shuffle_split(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    mask_dir = tmp_path / "masks"
    data_dir.mkdir()
    mask_dir.mkdir()
    (data_dir / "train-00000-of-00001.parquet").touch()
    np.save(mask_dir / "mask_7.npy", np.ones((2, 2), dtype=np.float32))
    items = [
        {
            "lensless": np.full((300, 400, 3), index, dtype=np.uint8),
            "lensed": np.full((20, 30, 3), index, dtype=np.uint8),
            "mask_label": 7,
        }
        for index in range(10)
    ]
    datasets_module = SimpleNamespace(
        load_dataset=lambda *args, **kwargs: FakeDataset(items)
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setattr(
        PSFStore,
        "_build",
        lambda self, mask_id: torch.ones(3, 300, 400),
    )

    train = DigiCamDataset(
        "train",
        data_dir=tmp_path,
        save_psf=False,
        limit=8,
        shuffle_seed=42,
    )
    validation = DigiCamDataset(
        "train",
        data_dir=tmp_path,
        save_psf=False,
        offset=8,
        limit=2,
        shuffle_seed=42,
    )

    assert set(train.indices).isdisjoint(validation.indices)
    assert set(train.indices) | set(validation.indices) == set(range(10))
