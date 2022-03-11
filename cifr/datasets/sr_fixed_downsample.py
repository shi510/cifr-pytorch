import math
import random

import torch
import numpy as np

from torchvision import transforms

from ..models.arch.utils import to_pixel_samples
from ..models.builder import DATASETS
from ..models.builder import build_dataset

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

@DATASETS.register_module()
class SRFixedDownsampled(torch.utils.data.Dataset):

    def __init__(self, dataset, inp_size, scale):
        self.dataset = build_dataset(dataset)
        self.inp_size = inp_size
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img = transforms.ToTensor()(img)

        w_lr = self.inp_size
        w_hr = round(w_lr * self.scale)

        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        real = img[:, x0: x0 + w_hr, y0: y0 + w_hr]

        fake_lr = resize_fn(real, w_lr)

        coord, _ = to_pixel_samples(real.contiguous())

        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / real.shape[-2]
        cell[:, 1] *= 2 / real.shape[-1]

        return {
            'lr': fake_lr,
            'coord': coord,
            'cell': cell,
            'real': real
        }
