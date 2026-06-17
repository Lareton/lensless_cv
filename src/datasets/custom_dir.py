from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.datasets.preprocessing import crop_roi, prepare_lensless, prepare_target
from src.datasets.psf import PSFStore

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class CustomDirDataset(Dataset):
    def __init__(self, data_dir, save_psf=True, psf_cache_dir=None, limit=None):
        self.root = Path(data_dir)
        self.lensless_dir = self.root / "lensless"
        self.lensed_dir = self.root / "lensed"
        self.mask_dir = self.root / "masks"

        if not self.lensless_dir.is_dir():
            raise FileNotFoundError(self.lensless_dir)
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(self.mask_dir)

        self.items = self._build_index()
        if limit is not None:
            self.items = self.items[:limit]

        cache_dir = psf_cache_dir or self.root / "psf"
        self.psf_store = PSFStore(
            self.mask_dir,
            cache_dir,
            save_psf,
            mask_pattern="{}.npy",
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        lensless = prepare_lensless(self._load_image(item["lensless"]))
        result = {
            "lensless": lensless,
            "psf": self.psf_store(item["image_id"]),
            "image_id": item["image_id"],
        }

        if item["lensed"] is not None:
            target = prepare_target(
                self._load_image(item["lensed"]),
                lensless.shape[-2:],
            )
            result["target"] = target
            result["target_roi"] = crop_roi(target)

        return result

    def _build_index(self):
        items = []
        for path in sorted(self.lensless_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_id = path.stem
            mask_path = self.mask_dir / f"{image_id}.npy"
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            lensed_path = self._find_image(self.lensed_dir, image_id)
            items.append(
                {
                    "image_id": image_id,
                    "lensless": path,
                    "lensed": lensed_path,
                }
            )

        if not items:
            raise ValueError(f"No images found in {self.lensless_dir}")

        return items

    @staticmethod
    def _find_image(directory, image_id):
        if not directory.is_dir():
            return None
        for extension in IMAGE_EXTENSIONS:
            path = directory / f"{image_id}{extension}"
            if path.exists():
                return path
        return None

    @staticmethod
    def _load_image(path):
        with Image.open(path) as image:
            return image.copy()
