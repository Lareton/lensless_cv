import numpy as np
import torch

from src.datasets.psf import PSFStore


def test_psf_store_saves_and_reuses_cache(tmp_path, monkeypatch):
    mask_dir = tmp_path / "masks"
    cache_dir = tmp_path / "psf"
    mask_dir.mkdir()
    np.save(mask_dir / "mask_4.npy", np.ones((2, 2), dtype=np.float32))

    calls = []

    def build(self, mask_id):
        calls.append(mask_id)
        return torch.ones(3, 8, 9)

    monkeypatch.setattr(PSFStore, "_build", build)

    first_store = PSFStore(mask_dir, cache_dir, save_to_disk=True)
    first = first_store(4)
    second = first_store(4)

    assert calls == ["4"]
    assert torch.equal(first, second)
    assert (cache_dir / "psf_4.pt").exists()

    monkeypatch.setattr(
        PSFStore,
        "_build",
        lambda self, mask_id: (_ for _ in ()).throw(AssertionError),
    )

    second_store = PSFStore(mask_dir, cache_dir, save_to_disk=True)
    cached = second_store(4)

    assert torch.equal(first, cached)


def test_psf_store_custom_mask_pattern(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    np.save(mask_dir / "sample.npy", np.ones((2, 2), dtype=np.float32))

    store = PSFStore(mask_dir, save_to_disk=False, mask_pattern="{}.npy")

    assert store.mask_pattern.format("sample") == "sample.npy"
