"""
DCGAN (Deep Convolutional Generative Adversarial Network) for animal image generation.

Architecture follows Radford et al. (2015) with modifications for stable training:
- Generator: transposed convolutions from latent z to 64x64 RGB image
- Discriminator: strided convolutions from 64x64 RGB image to real/fake prediction
- BatchNorm in both networks (except output layers)
- LeakyReLU in discriminator, ReLU in generator
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Maps a latent vector z (nz,) to a 64x64 RGB image via fractional-strided convolutions."""

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        # nz: latent vector dimension
        # ngf: base feature map size in generator
        # nc: number of output channels (3 for RGB)
        self.main = nn.Sequential(
            # (nz) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8, 4, 4) -> (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4, 8, 8) -> (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2, 16, 16) -> (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf, 32, 32) -> (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    """Classifies 64x64 RGB images as real or fake via strided convolutions."""

    def __init__(self, nc: int = 3, ndf: int = 64):
        super().__init__()
        # nc: number of input channels
        # ndf: base feature map size in discriminator
        self.main = nn.Sequential(
            # (nc, 64, 64) -> (ndf, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf, 32, 32) -> (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2, 16, 16) -> (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4, 8, 8) -> (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8, 4, 4) -> (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1).squeeze(1)


def weights_init(m: nn.Module):
    """Custom weight initialization as per DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
