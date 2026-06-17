import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualStack(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

    def forward(self, x):
        return self.blocks(x)


class DRUNet(nn.Module):
    def __init__(
        self,
        channels=(32, 64, 128, 256),
        blocks=4,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("DRUNet needs four channel scales")

        c1, c2, c3, c4 = channels
        self.gradient_checkpointing = gradient_checkpointing
        self.head = nn.Conv2d(3, c1, 3, padding=1)
        self.down_blocks = nn.ModuleList(
            (
                ResidualStack(c1, blocks),
                ResidualStack(c2, blocks),
                ResidualStack(c3, blocks),
            )
        )
        self.downsamples = nn.ModuleList(
            (
                nn.Conv2d(c1, c2, 2, stride=2),
                nn.Conv2d(c2, c3, 2, stride=2),
                nn.Conv2d(c3, c4, 2, stride=2),
            )
        )
        self.body = ResidualStack(c4, blocks)
        self.upsamples = nn.ModuleList(
            (
                nn.ConvTranspose2d(c4, c3, 2, stride=2),
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
            )
        )
        self.up_blocks = nn.ModuleList(
            (
                ResidualStack(c3, blocks),
                ResidualStack(c2, blocks),
                ResidualStack(c1, blocks),
            )
        )
        self.tail = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

        x1 = self.head(x)
        x2 = self.downsamples[0](self._run(self.down_blocks[0], x1))
        x3 = self.downsamples[1](self._run(self.down_blocks[1], x2))
        x4 = self.downsamples[2](self._run(self.down_blocks[2], x3))

        x = self._run(self.body, x4)
        x = self.upsamples[0](x + x4)
        x = self._run(self.up_blocks[0], x)
        x = self.upsamples[1](x + x3)
        x = self._run(self.up_blocks[1], x)
        x = self.upsamples[2](x + x2)
        x = self._run(self.up_blocks[2], x)
        x = self.tail(x + x1)

        return x[..., :height, :width]

    def _run(self, module, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)
