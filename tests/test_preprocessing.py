import numpy as np
import torch

from src.datasets.preprocessing import crop_roi, prepare_lensless, prepare_target


def test_prepare_lensless():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[0, 0] = 255

    result = prepare_lensless(image)

    assert result.shape == (3, 300, 400)
    assert result.dtype == torch.float32
    assert torch.all(result[:, -1, -1] == 1)
    assert result.min() == 0
    assert result.max() == 1


def test_prepare_target_and_roi():
    image = np.full((20, 30, 3), 255, dtype=np.uint8)

    target = prepare_target(image, (300, 400))
    roi = crop_roi(target)

    assert target.shape == (3, 300, 400)
    assert roi.shape == (3, 200, 266)
    assert torch.all(roi == 1)
    assert target.sum() == roi.sum()
