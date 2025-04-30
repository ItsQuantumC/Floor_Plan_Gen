import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, act='relu', dropout=False):
        super().__init__()
        self.down = down
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False) if down else
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(0.2)
        )
        self.drop = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        return self.drop(self.block(x))

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 64,  down=True,  act='leaky')
        self.enc2 = UNetBlock(64, 128,   down=True,  act='leaky')
        self.enc3 = UNetBlock(128, 256,  down=True,  act='leaky')
        self.enc4 = UNetBlock(256, 512,  down=True,  act='leaky')
        self.enc5 = UNetBlock(512, 512,  down=True,  act='leaky')
        self.enc6 = UNetBlock(512, 512,  down=True,  act='leaky')
        self.enc7 = UNetBlock(512, 512,  down=True,  act='leaky')
        self.bottleneck = UNetBlock(512, 512, down=True)

        self.dec1 = UNetBlock(512, 512, down=False, dropout=True)
        self.dec2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.dec3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.dec4 = UNetBlock(1024, 512, down=False)
        self.dec5 = UNetBlock(1024, 256, down=False)
        self.dec6 = UNetBlock(512, 128, down=False)
        self.dec7 = UNetBlock(256, 64,  down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        e4 = self.enc4(e3); e5 = self.enc5(e4); e6 = self.enc6(e5)
        e7 = self.enc7(e6); b = self.bottleneck(e7)

        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e7], 1))
        d3 = self.dec3(torch.cat([d2, e6], 1))
        d4 = self.dec4(torch.cat([d3, e5], 1))
        d5 = self.dec5(torch.cat([d4, e4], 1))
        d6 = self.dec6(torch.cat([d5, e3], 1))
        d7 = self.dec7(torch.cat([d6, e2], 1))
        return self.final(torch.cat([d7, e1], 1))
