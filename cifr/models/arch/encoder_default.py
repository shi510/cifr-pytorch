import math

import torch.nn as nn
import torch.nn.functional as F

from ..builder import ARCHITECTURES


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.LReLU1 = nn.LeakyReLU(0.2, inplace=not residual)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.LReLU2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.LReLU3 = nn.LeakyReLU(0.2, inplace=True)
        self.residual = residual

    def forward(self, x):
        x = shortcut = self.conv1(x)
        x = self.LReLU1(x)
        x = self.conv2(x)
        x = self.LReLU2(x)
        x = self.conv3(x)
        x = self.LReLU3(x)
        if self.residual:
            x = x + shortcut
        return x

@ARCHITECTURES.register_module()
class EncoderDefault(nn.Module):
    def __init__(self, downscale, latent_dim, mode="max"):
        super(EncoderDefault, self).__init__()
        downstep = int(math.log(downscale, 2))
        feat_spec = {
            0: 32,
            1: 64,
            2: 128,
            3: 256,
            4: 512
        }
        self.out_channels = feat_spec[downstep]

        self.convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = 3
        for i in range(downstep):
            self.convs.append(Conv2dBlock(in_ch, feat_spec[i], residual=True))
            self.convs.append(Conv2dBlock(feat_spec[i], feat_spec[i], residual=True))
            if mode == "max":
                ds = nn.MaxPool2d(2)
            elif mode == "avg":
                ds = nn.AvgPool2d(2, 2)
            else:
                ds = lambda x : F.interpolate(x, scale_factor=0.5, mode=mode, align_corners=True, recompute_scale_factor=True)
            self.downsamples.append(ds)
            in_ch = feat_spec[i]

        self.tail_convs = nn.ModuleList()
        self.tail_convs.append(Conv2dBlock(in_ch, feat_spec[downstep], residual=True))
        self.tail_convs.append(Conv2dBlock(feat_spec[downstep], feat_spec[downstep], residual=True))

        self.gavg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.final_linear = nn.Sequential(
            nn.Linear(feat_spec[downstep], latent_dim),
            nn.LeakyReLU(),
        )

    def get_output_channels(self):
        return self.out_channels

    def forward(self, x):
        interm_feat = []
        for conv1, conv2, downsample in zip(self.convs[::2], self.convs[1::2], self.downsamples):
            x = conv1(x)
            x = conv2(x)
            interm_feat.append(x)
            x = downsample(x)
        
        for conv1, conv2 in zip(self.tail_convs[::2], self.tail_convs[1::2]):
            x = conv1(x)
            x = conv2(x)
        interm_feat.append(x)
        interm_feat.reverse()
        x = self.gavg_pool(x).view(x.shape[0:2])
        x = self.final_linear(x)

        return x, interm_feat

if __name__ == '__main__':
    import torch
    m = EncoderDefault(downscale=8, latent_dim=1024).cuda()
    img = torch.randn(2, 3, 32, 32).cuda()
    output, features = m(img)
    print(output.shape, m.get_output_channels())
    for t in features:
        print(t.shape)
