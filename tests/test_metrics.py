from types import SimpleNamespace

import torch

import src.metrics.image as image_metrics
from src.metrics.evaluate import reconstruction_roi, target_roi


class DummyMetric:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self

    def __call__(self, prediction, target):
        return prediction.new_tensor(self.value)


def test_image_metric_tracker(monkeypatch):
    values = iter((DummyMetric(1), DummyMetric(0)))
    monkeypatch.setattr(image_metrics, "instantiate", lambda config: next(values))
    tracker = image_metrics.ImageMetricTracker(
        SimpleNamespace(ssim=None, lpips=None),
        "cpu",
    )
    image = torch.rand(2, 3, 32, 40)

    tracker.update(image, image)
    result = tracker.compute()

    assert result["mse"] == 0
    assert result["ssim"] == 1
    assert result["lpips"] == 0
    assert result["psnr"] > 60


def test_metric_roi():
    reconstruction = torch.rand(3, 300, 400)
    target = torch.rand(3, 500, 500)

    prediction_roi = reconstruction_roi(reconstruction)
    resized_target = target_roi(target, reconstruction.shape[-2:])

    assert prediction_roi.shape == (3, 200, 266)
    assert resized_target.shape == (3, 200, 266)
