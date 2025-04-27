import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1)]
            if norm: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels * 2, 64, norm=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))
