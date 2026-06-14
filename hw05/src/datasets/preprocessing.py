import cv2
import numpy as np
import torch

ROI_TOP = 80
ROI_LEFT = 100
ROI_HEIGHT = 200
ROI_WIDTH = 266


def force_rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    return image


def image_to_float(image):
    image = force_rgb(image)
    if np.issubdtype(image.dtype, np.integer):
        max_value = np.iinfo(image.dtype).max
        image = image.astype(np.float32) / max_value
    else:
        image = image.astype(np.float32)
    return image


def prepare_lensless(image):
    image = image_to_float(image)
    image = torch.from_numpy(image)
    image = torch.rot90(image, dims=(0, 1), k=2)
    return image.permute(2, 0, 1).contiguous()


def prepare_target(image, shape):
    image = image_to_float(image)
    image = cv2.resize(
        image,
        dsize=(ROI_WIDTH, ROI_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    target = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    target[
        ROI_TOP : ROI_TOP + ROI_HEIGHT,
        ROI_LEFT : ROI_LEFT + ROI_WIDTH,
    ] = image
    return torch.from_numpy(target).permute(2, 0, 1).contiguous()


def crop_roi(image):
    return image[
        ...,
        ROI_TOP : ROI_TOP + ROI_HEIGHT,
        ROI_LEFT : ROI_LEFT + ROI_WIDTH,
    ]


def prepare_psf(psf):
    if psf.ndim == 4:
        psf = psf.squeeze(0)
    if psf.ndim != 3:
        raise ValueError(f"Expected a HWC PSF, got {tuple(psf.shape)}")
    return psf.permute(2, 0, 1).float().contiguous()
