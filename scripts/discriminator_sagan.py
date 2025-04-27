import torch
import torch.nn as nn
from self_attention import SelfAttention

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, fmap=64):
        super().__init__()
        self.model = nn.Sequential(
            # 128×128 → 64×64
            nn.Conv2d(img_channels, fmap, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64×64 → 32×32
            nn.Conv2d(fmap, fmap*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap*2), nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(fmap*2),

            # 32×32 → 16×16
            nn.Conv2d(fmap*2, fmap*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap*4), nn.LeakyReLU(0.2, inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(fmap*4, fmap*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap*8), nn.LeakyReLU(0.2, inplace=True),

            # final patch output
            nn.Conv2d(fmap*8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img):
        out = self.model(img)
        return out.view(out.size(0), -1)
