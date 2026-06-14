import torch
from hydra.utils import instantiate


class ImageMetricTracker:
    def __init__(self, config, device):
        self.device = device
        self.ssim = instantiate(config.ssim).to(device)
        self.lpips = instantiate(config.lpips).to(device)
        self.reset()

    def reset(self):
        self.total = {
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "lpips": 0.0,
        }
        self.count = 0

    @torch.no_grad()
    def update(self, prediction, target):
        prediction = prediction.clamp(0, 1)
        target = target.clamp(0, 1)
        batch_size = prediction.shape[0]
        mse = (prediction - target).square().flatten(1).mean(dim=1)
        psnr = -10 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).eps))

        for index in range(batch_size):
            current_prediction = prediction[index : index + 1]
            current_target = target[index : index + 1]
            self.total["mse"] += mse[index].item()
            self.total["psnr"] += psnr[index].item()
            self.total["ssim"] += self.ssim(
                current_prediction,
                current_target,
            ).item()
            self.total["lpips"] += self.lpips(
                current_prediction,
                current_target,
            ).item()

        self.count += batch_size

    def compute(self):
        if self.count == 0:
            return {}
        return {name: value / self.count for name, value in self.total.items()}
