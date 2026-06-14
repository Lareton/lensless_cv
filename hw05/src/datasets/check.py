from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate


def normalize(image):
    image = image.float()
    return image / image.amax().clamp_min(1e-8)


def check_dataset(config):
    dataset = instantiate(config.dataset)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item = dataset[config.index]
    images = [
        ("lensless", item["lensless"]),
        ("target", item.get("target")),
        ("target_roi", item.get("target_roi")),
        ("psf", normalize(item["psf"])),
    ]

    images = [(name, image) for name, image in images if image is not None]
    figure, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for axis, (name, image) in zip(axes, images):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(f"{name} {tuple(image.shape)}")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_dir / f"{item['image_id']}.png", dpi=150)
    plt.close(figure)

    print(
        {
            key: tuple(value.shape)
            for key, value in item.items()
            if torch.is_tensor(value)
        }
    )
