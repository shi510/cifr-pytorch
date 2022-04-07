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
from cifr.models.builder import build_architecture
from cifr.models.arch.utils import make_coord
from tools.utils import query_all_pixels
from tools.utils import add_img_plot


WORK_DIR = './work_dir'

def generate_continuous_zoomin_bboxes(w, h, roi):
    x1, y1, x2, y2 = roi

    lt_a = y1/x1
    rb_a = (h-y2)/(w-x2)

    lt_d = math.sqrt(y1**2+x1**2)
    rb_d = math.sqrt((h-y2)**2+(w-x2)**2)

    max_idx = np.array([lt_d, rb_d]).argmax()
    x_range = np.array([x1, w-x2], dtype=np.int32)[max_idx]

    x1s = np.linspace(0, x1, x_range)
    y1s = np.clip(x1s*lt_a, 0, h-1)
    x2s = np.linspace(x2, w, x_range)[::-1]
    y2s = np.clip(x2s*rb_a + h*(1-rb_a), 0, h-1)
    return np.stack([x1s, y1s, x2s, y2s], axis=1).astype(np.uint32)

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

def inference(args, config):
    encoder = build_architecture(config.encoder).cuda()
    model = build_architecture(config.model).cuda()
    state_dict = torch.load(args.ckpt)
    encoder_ema = ExponentialMovingAverage(encoder.parameters(), decay=0.995)
    model_ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    encoder_ema.load_state_dict(state_dict['encoder_ema'])
    model_ema.load_state_dict(state_dict['model_ema'])
    encoder_ema.copy_to(encoder.parameters())
    model_ema.copy_to(model.parameters())
    encoder.eval()
    model.eval()

    img = Image.open(args.img)
    img = img.convert("RGB")
    img = transforms.ToTensor()(img)
    lr_img = resize_fn(img, (64, 64)).unsqueeze(0).cuda()
    lr_img = (lr_img - 0.5) / 0.5

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    os.makedirs(f'{WORK_DIR}/{config_name}/eval', exist_ok=True)
    upsample_list = [2, 3, 4, 8, 16]
    rows = 2
    cols = len(upsample_list) + 1
    fig = plt.figure(figsize=(15, rows*6))
    add_img_plot(fig, lr_img[0], f'Input', rows, cols, 1)
    for n, xup in enumerate([2, 3, 4, 8, 16]):
        target_H = lr_img.shape[2]*xup
        target_W = lr_img.shape[3]*xup
        coord = make_coord((target_H, target_W)).unsqueeze(0).cuda()
        cell = torch.ones_like(coord).cuda()
        cell[:, :, 0] *= 2 / (target_H)
        cell[:, :, 1] *= 2 / (target_W)
        pred = query_all_pixels(encoder, model, lr_img, coord, cell, 1024)
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
    with torch.no_grad():
        inference(args, cfg)
