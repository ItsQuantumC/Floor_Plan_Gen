import torch
import torch.nn as nn
from self_attention import SelfAttention

class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, fmap=64):
        super().__init__()
        self.init_size = 8  # 8×8 start
        self.l1 = nn.Linear(z_dim, fmap*8*self.init_size*self.init_size)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(fmap*8),
            # 8×8 → 16×16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(fmap*8, fmap*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fmap*4), nn.ReLU(inplace=True),

            SelfAttention(fmap*4),

            # 16×16 → 32×32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(fmap*4, fmap*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fmap*2), nn.ReLU(inplace=True),

            # 32×32 → 64×64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(fmap*2, fmap, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fmap), nn.ReLU(inplace=True),

            # 64×64 → 128×128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(fmap, img_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(z.size(0), -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
