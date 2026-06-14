from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.datasets.preprocessing import (
    ROI_HEIGHT,
    ROI_LEFT,
    ROI_TOP,
    ROI_WIDTH,
    crop_roi,
    image_to_float,
)
from src.metrics.image import ImageMetricTracker

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_index(directory):
    directory = Path(directory)
    if (directory / "lensed").is_dir():
        directory = directory / "lensed"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }


def load_image(path):
    with Image.open(path) as image:
        array = image_to_float(image.copy())
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def reconstruction_roi(image):
    if image.shape[-2:] == (ROI_HEIGHT, ROI_WIDTH):
        return image
    if (
        image.shape[-2] >= ROI_HEIGHT + ROI_TOP
        and image.shape[-1] >= ROI_WIDTH + ROI_LEFT
    ):
        return crop_roi(image)
    raise ValueError(f"Invalid reconstruction shape {tuple(image.shape)}")


def target_roi(image, reconstruction_shape):
    if image.shape[-2:] == reconstruction_shape:
        return reconstruction_roi(image)
    if image.shape[-2:] == (ROI_HEIGHT, ROI_WIDTH):
        return image

    array = image.permute(1, 2, 0).numpy()
    resized = Image.fromarray(np.round(array * 255).astype(np.uint8)).resize(
        (ROI_WIDTH, ROI_HEIGHT),
        resample=Image.Resampling.NEAREST,
    )
    return load_pil(resized)


def load_pil(image):
    array = image_to_float(image)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def evaluate_directories(
    ground_truth_dir,
    reconstruction_dir,
    metric_config,
    device,
):
    ground_truth = image_index(ground_truth_dir)
    reconstructions = image_index(reconstruction_dir)
    matched = sorted(ground_truth.keys() & reconstructions.keys())

    if not matched:
        raise ValueError("No matching image ids")

    tracker = ImageMetricTracker(metric_config, device)
    for image_id in matched:
        reconstruction = load_image(reconstructions[image_id])
        prediction = reconstruction_roi(reconstruction)
        target = target_roi(
            load_image(ground_truth[image_id]),
            reconstruction.shape[-2:],
        )
        tracker.update(
            prediction.unsqueeze(0).to(device),
            target.unsqueeze(0).to(device),
        )

    result = tracker.compute()
    result["matched"] = len(matched)
    result["missing_ground_truth"] = len(reconstructions.keys() - ground_truth.keys())
    result["missing_reconstructions"] = len(
        ground_truth.keys() - reconstructions.keys()
    )
    return result
