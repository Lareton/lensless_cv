import numpy as np
import torch
from PIL import Image

from src.datasets import CustomDirDataset, collate_fn
from src.datasets.psf import PSFStore


def save_image(path, value):
    image = np.full((300, 400, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_custom_dir_with_target(tmp_path, monkeypatch):
    lensless_dir = tmp_path / "lensless"
    lensed_dir = tmp_path / "lensed"
    mask_dir = tmp_path / "masks"
    lensless_dir.mkdir()
    lensed_dir.mkdir()
    mask_dir.mkdir()

    save_image(lensless_dir / "first.png", 32)
    save_image(lensed_dir / "first.png", 64)
    np.save(mask_dir / "first.npy", np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr(
        PSFStore,
        "_build",
        lambda self, mask_id: torch.ones(3, 300, 400),
    )

    dataset = CustomDirDataset(tmp_path, save_psf=False)
    item = dataset[0]

    assert item["image_id"] == "first"
    assert item["lensless"].shape == (3, 300, 400)
    assert item["target"].shape == (3, 300, 400)
    assert item["target_roi"].shape == (3, 200, 266)
    assert item["psf"].shape == (3, 300, 400)


def test_custom_dir_without_target(tmp_path, monkeypatch):
    lensless_dir = tmp_path / "lensless"
    mask_dir = tmp_path / "masks"
    lensless_dir.mkdir()
    mask_dir.mkdir()

    for image_id in ("first", "second"):
        save_image(lensless_dir / f"{image_id}.png", 32)
        np.save(mask_dir / f"{image_id}.npy", np.ones((2, 2), dtype=np.float32))

    monkeypatch.setattr(
        PSFStore,
        "_build",
        lambda self, mask_id: torch.ones(3, 300, 400),
    )

    dataset = CustomDirDataset(tmp_path, save_psf=False)
    batch = collate_fn([dataset[0], dataset[1]])

    assert "target" not in batch
    assert batch["lensless"].shape == (2, 3, 300, 400)
    assert batch["psf"].shape == (2, 3, 300, 400)
    assert batch["image_id"] == ["first", "second"]
