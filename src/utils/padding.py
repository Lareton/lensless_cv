import math

import torch.nn.functional as F


def next_fast_size(size):
    size = int(size)
    while True:
        value = size
        for factor in (2, 3, 5):
            while value % factor == 0:
                value //= factor
        if value == 1:
            return size
        size += 1


def fft_shape(shape, scale=2):
    height, width = shape[-2:]
    return next_fast_size(math.ceil(height * scale)), next_fast_size(
        math.ceil(width * scale)
    )


def center_pad(x, shape):
    height, width = x.shape[-2:]
    target_height, target_width = shape

    if target_height < height or target_width < width:
        raise ValueError(f"Cannot pad {x.shape[-2:]} to {shape}")

    pad_height = target_height - height
    pad_width = target_width - width
    top = target_height // 2 - height // 2
    left = target_width // 2 - width // 2

    return F.pad(
        x,
        (
            left,
            pad_width - left,
            top,
            pad_height - top,
        ),
    )


def center_crop(x, shape):
    height, width = x.shape[-2:]
    target_height, target_width = shape

    if target_height > height or target_width > width:
        raise ValueError(f"Cannot crop {x.shape[-2:]} to {shape}")

    top = height // 2 - target_height // 2
    left = width // 2 - target_width // 2

    return x[..., top : top + target_height, left : left + target_width]
