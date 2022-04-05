import argparse
import os
import math

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch_ema import ExponentialMovingAverage

from cifr.core.config import Config
from cifr.models.builder import build_generator
from cifr.models.arch.utils import make_coord


WORK_DIR = './work_dir'

def add_img_plot(fig, img, title, rows, cols, num):
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose((1, 2, 0))
    ax = fig.add_subplot(rows, cols, num)
    ax.imshow(img)
    ax.set_title(title, size=25)
    ax.axis("off")

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

def inference(args, config):
    gen = build_generator(config.generator).cuda()
    state_dict = torch.load(args.ckpt)
    encoder_ema = ExponentialMovingAverage(gen.parameters(), decay=0.995)
    encoder_ema.load_state_dict(state_dict['generator_ema'])
    encoder_ema.copy_to(gen.parameters())
    gen.eval()

    img = Image.open(args.img)
    img = img.convert("RGB")
    img = transforms.ToTensor()(img)
    lr_img = resize_fn(img, (64, 64)).unsqueeze(0).cuda()

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs(f'{WORK_DIR}/{config_name}/eval', exist_ok=True)
    upsample_list = [2, 3, 4, 8, 16, 32]
    rows = 2
    cols = len(upsample_list) + 1
    fig = plt.figure(figsize=(15, rows*6))
    add_img_plot(fig, lr_img[0], f'Input', rows, cols, 1)
    for n, xup in enumerate(upsample_list):
        target_H = lr_img.shape[2]*xup
        target_W = lr_img.shape[3]*xup
        coord = make_coord((target_H, target_W)).unsqueeze(0).cuda()
        cell = torch.ones_like(coord).cuda()
        cell[:, :, 0] *= 2 / (target_H)
        cell[:, :, 1] *= 2 / (target_W)
        with torch.no_grad():
            pred = gen(lr_img, coord, cell)
        add_img_plot(fig, pred[0], f'Up x{xup}', rows, cols, n + 2)

    file_name = os.path.splitext(os.path.basename(args.img))[0]
    plt.tight_layout()
    plt.savefig(f'{WORK_DIR}/{config_name}/eval/{file_name}.jpg', bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img', type=str, required=True)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    inference(args, cfg)
