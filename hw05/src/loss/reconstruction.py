import torch.nn.functional as F
from torch import nn

from src.datasets.preprocessing import crop_roi


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        mse_weight=(1.0, 0.1),
        lpips_weight=(0.0, 1.0),
        lpips_net="vgg",
    ):
        super().__init__()
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=lpips_net,
            normalize=True,
        )
        self.progress = 0.0

    def set_progress(self, progress):
        self.progress = min(max(float(progress), 0.0), 1.0)

    def forward(self, reconstruction, target_roi):
        prediction = crop_roi(reconstruction).clamp(0, 1)
        target = target_roi.clamp(0, 1)
        mse = F.mse_loss(prediction, target)
        lpips = self.lpips(prediction, target).mean()
        mse_weight = self._weight(self.mse_weight)
        lpips_weight = self._weight(self.lpips_weight)

        return {
            "loss": mse_weight * mse + lpips_weight * lpips,
            "mse": mse,
            "lpips": lpips,
            "mse_weight": prediction.new_tensor(mse_weight),
            "lpips_weight": prediction.new_tensor(lpips_weight),
        }

    def _weight(self, values):
        start, end = values
        return start + self.progress * (end - start)
