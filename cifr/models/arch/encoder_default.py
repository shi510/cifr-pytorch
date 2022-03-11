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
    def __init__(self, downsample="max"):
        super(EncoderDefault, self).__init__()

        self.conv1_1 = Conv2dBlock(3, 32, residual=True)
        self.conv1_2 = Conv2dBlock(32, 32, residual=True)
        if downsample == "max":
            self.downsample1 = nn.MaxPool2d(2)
        else:
            self.downsample1 = lambda x : F.interpolate(x, scale_factor=0.5, mode=downsample, align_corners=True, recompute_scale_factor=True)

        self.conv2_1 = Conv2dBlock(32, 64, residual=True)
        self.conv2_2 = Conv2dBlock(64, 64, residual=True)
        if downsample == "max":
            self.downsample2 = nn.MaxPool2d(2)
        else:
            self.downsample2 = lambda x : F.interpolate(x, scale_factor=0.5, mode=downsample, align_corners=True, recompute_scale_factor=True)

        self.conv3_1 = Conv2dBlock(64, 128, residual=True)
        self.conv3_2 = Conv2dBlock(128, 128, residual=True)
        if downsample == "max":
            self.downsample3 = nn.MaxPool2d(2)
        else:
            self.downsample3 = lambda x : F.interpolate(x, scale_factor=0.5, mode=downsample, align_corners=True, recompute_scale_factor=True)

        self.conv4_1 = Conv2dBlock(128, 256, residual=True)
        self.conv4_2 = Conv2dBlock(256, 256, residual=True)
        if downsample == "max":
            self.downsample4 = nn.MaxPool2d(2)
        else:
            self.downsample4 = lambda x : F.interpolate(x, scale_factor=0.5, mode=downsample, align_corners=True, recompute_scale_factor=True)

        self.conv5_1 = Conv2dBlock(256, 512, residual=True)
        self.conv5_2 = Conv2dBlock(512, 512, residual=True)

        self.gavg_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        feat1 = x
        x = self.downsample1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        feat2 = x
        x = self.downsample2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        feat3 = x
        x = self.downsample3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        feat4 = x
        x = self.downsample4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        feat5 = x

        return x, [feat5, feat4, feat3, feat2, feat1]
