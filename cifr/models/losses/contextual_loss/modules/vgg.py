from collections import namedtuple

import torch.nn as nn
import torchvision.models.vgg as vgg


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 8):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 17):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 26):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(26, 35):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for x in range(35, 37):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_conv1_2 = h
        h = self.slice2(h)
        h_conv2_2 = h
        h = self.slice3(h)
        h_conv3_4 = h
        h = self.slice4(h)
        h_conv4_4 = h
        h = self.slice5(h)
        h_conv5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['conv1_2', 'conv2_2',
                           'conv3_4', 'conv4_4', 'conv5_4'])
        out = vgg_outputs(h_conv1_2, h_conv2_2,
                          h_conv3_4, h_conv4_4, h_conv5_4)

        return out
