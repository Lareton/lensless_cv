from pathlib import Path

REPO_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


def split_files(root, split):
    return sorted((Path(root) / "data").glob(f"{split}-*.parquet"))


def dataset_exists(root, split):
    root = Path(root)
    return bool(split_files(root, split)) and any((root / "masks").glob("mask_*.npy"))


def download_digicam(output_dir, repo_id=REPO_ID, split=None):
    from huggingface_hub import snapshot_download

    patterns = ["masks/*.npy"]
    if split is None:
        patterns.append("data/*.parquet")
    else:
        patterns.append(f"data/{split}-*.parquet")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=patterns,
    )

    return Path(output_dir)


def ensure_digicam(root, split, download_if_missing, repo_id=REPO_ID):
    root = Path(root)
    if dataset_exists(root, split):
        return root
    if download_if_missing:
        return download_digicam(root, repo_id, split)
    raise FileNotFoundError(
        f"DigiCam {split} split was not found in {root}. "
        f"Run python download_dataset.py split={split}"
    )
