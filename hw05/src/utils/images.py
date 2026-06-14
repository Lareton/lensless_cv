from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.datasets.preprocessing import ROI_HEIGHT, ROI_WIDTH, crop_roi


def normalize_image(image):
    image = image.float()
    minimum = image.amin(dim=(-2, -1), keepdim=True)
    maximum = image.amax(dim=(-2, -1), keepdim=True)
    return (image - minimum) / (maximum - minimum).clamp_min(1e-8)


def tensor_to_image(image):
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8))


def panel(image, title, normalize=False):
    if normalize:
        image = normalize_image(image)
    image = tensor_to_image(image)
    image = image.resize((ROI_WIDTH, ROI_HEIGHT), Image.Resampling.BILINEAR)
    result = Image.new("RGB", (ROI_WIDTH, ROI_HEIGHT + 24), "white")
    result.paste(image, (0, 24))
    ImageDraw.Draw(result).text((8, 5), title, fill="black")
    return result


def save_qualitative_grid(batch, reconstruction, output_dir, limit):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(limit, reconstruction.shape[0])
    paths = []

    for index in range(count):
        panels = [
            panel(batch["lensless"][index], "lensless", normalize=True),
            panel(
                crop_roi(reconstruction[index]),
                "reconstruction",
            ),
            panel(batch["psf"][index], "psf", normalize=True),
        ]

        if "target_roi" in batch:
            panels.insert(0, panel(batch["target_roi"][index], "ground truth"))

        grid = Image.new(
            "RGB",
            (ROI_WIDTH * len(panels), ROI_HEIGHT + 24),
            "white",
        )
        for panel_index, current_panel in enumerate(panels):
            grid.paste(current_panel, (panel_index * ROI_WIDTH, 0))

        path = output_dir / f"{batch['image_id'][index]}.png"
        grid.save(path)
        paths.append(path)

    return paths
