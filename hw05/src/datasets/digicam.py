from torch.utils.data import Dataset

from src.datasets.download import REPO_ID, ensure_digicam, split_files
from src.datasets.preprocessing import crop_roi, prepare_lensless, prepare_target
from src.datasets.psf import PSFStore


class DigiCamDataset(Dataset):
    def __init__(
        self,
        split,
        data_dir="data/digicam",
        download_if_missing=False,
        save_psf=True,
        psf_cache_dir=None,
        offset=0,
        limit=None,
        repo_id=REPO_ID,
    ):
        from datasets import load_dataset

        self.root = ensure_digicam(data_dir, split, download_if_missing, repo_id)
        self.split = split

        files = [str(path) for path in split_files(self.root, split)]
        self.dataset = load_dataset("parquet", data_files={split: files}, split=split)

        end = (
            len(self.dataset)
            if limit is None
            else min(offset + limit, len(self.dataset))
        )
        self.indices = range(offset, end)
        self.dataset = self.dataset.select(self.indices)

        cache_dir = psf_cache_dir or self.root / "psf"
        self.psf_store = PSFStore(self.root / "masks", cache_dir, save_psf)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        lensless = prepare_lensless(item["lensless"])
        target = prepare_target(item["lensed"], lensless.shape[-2:])

        return {
            "lensless": lensless,
            "target": target,
            "target_roi": crop_roi(target),
            "psf": self.psf_store(item["mask_label"]),
            "mask_label": item["mask_label"],
            "image_id": f"{self.split}_{self.indices[index]:05d}",
        }
