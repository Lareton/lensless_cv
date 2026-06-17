import os
from pathlib import Path

import numpy as np
import torch


class PSFStore:
    def __init__(
        self,
        mask_dir,
        cache_dir=None,
        save_to_disk=True,
        mask_pattern="mask_{}.npy",
    ):
        self.mask_dir = Path(mask_dir)
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else self.mask_dir.parent / "psf"
        )
        self.save_to_disk = save_to_disk
        self.mask_pattern = mask_pattern
        self.cache = {}

        if self.save_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, mask_id):
        mask_id = str(mask_id)
        if mask_id in self.cache:
            return self.cache[mask_id]

        cache_path = self.cache_dir / f"psf_{mask_id}.pt" if self.save_to_disk else None
        if cache_path is not None and cache_path.exists():
            psf = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            psf = self._build(mask_id)
            if cache_path is not None:
                self._save(psf, cache_path)

        self.cache[mask_id] = psf
        return psf

    def _build(self, mask_id):
        from lensless_helpers.psf import simulate_psf_from_mask

        mask_path = self.mask_dir / self.mask_pattern.format(mask_id)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        mask = np.load(mask_path)
        psf = simulate_psf_from_mask(mask)

        if psf.ndim == 4:
            psf = psf.squeeze(0)

        return psf.permute(2, 0, 1).float().contiguous()

    @staticmethod
    def _save(psf, path):
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(psf, tmp_path)
        if path.exists():
            tmp_path.unlink()
        else:
            tmp_path.replace(path)
