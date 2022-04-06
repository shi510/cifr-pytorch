import math

import torch.nn as nn

from ..builder import ARCHITECTURES
from ..builder import build_architecture
from .stylegan_utils import StyledConv, ToRGB, ConstantInput


@ARCHITECTURES.register_module()
class StyleGAN2(nn.Module):
    def __init__(self, arch, size=128, style_dim=512, rgb_dim=3, activation=None):
        super(StyleGAN2, self).__init__()

        self.style_dim = style_dim
        self.backbone = build_architecture(arch)

        self.size = size
        self.demodulate = True

        self.channels = {
            4: 512,
            8: 256,
            16: 128,
            32: 64,
            64: 32,
        }

        self.constant4x4 = ConstantInput(self.channels[4], 4)

        self.lin0 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, demodulate=self.demodulate)
        self.to_rgb0 = ToRGB(self.channels[4], style_dim, rgb_dim=rgb_dim, upsample=False)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        in_channels = self.channels[4]
        for i in range(3, self.log_size+1):
            out_channel = self.channels[2 ** i]
            self.linears.append(StyledConv(in_channels, out_channel, 3, style_dim,
                                           demodulate=self.demodulate, activation=activation, upsample=True))
            self.linears.append(StyledConv(out_channel, out_channel, 3, style_dim,
                                           demodulate=self.demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channel, style_dim, rgb_dim=rgb_dim, upsample=True))

            in_channels = out_channel

    def forward(self, x):
        latent, features = self.backbone(x)
        out = features[0] + self.constant4x4(x.shape[0])
        out = self.lin0(out, latent)
        rgb = self.to_rgb0(out, latent)
        for lin1, lin2, to_rgb, feat in zip(self.linears[::2], self.linears[1::2], self.to_rgbs, features[1:]):
            out = lin1(out, latent) + feat
            out = lin2(out, latent)
            rgb = to_rgb(out, latent, rgb)
        return rgb
