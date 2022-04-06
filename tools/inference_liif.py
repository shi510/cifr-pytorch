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


WORK_DIR = './work_dir'

def add_img_plot(fig, img, title, rows, cols, num):
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose((1, 2, 0))
    ax = fig.add_subplot(rows, cols, num)
    ax.imshow(img)
    ax.set_title(title, size=25)
    ax.axis("off")

@torch.no_grad()
def query_all_pixels(encoder, model, lr, coord, cell, bsize):
    feature = encoder(lr)
    n = coord.shape[1]
    ql = 0
    preds = []
    while ql < n:
        qr = min(ql + bsize, n)
        pred = model(lr, feature, coord[:, ql: qr, :], cell[:, ql: qr, :])
        preds.append(pred)
        ql = qr
    pred = torch.cat(preds, dim=1)
    pred.clamp_(0, 1)
    ih, iw = lr.shape[-2:]
    s = math.sqrt(coord.shape[1] / (ih * iw))
    shape = [lr.shape[0], round(ih * s), round(iw * s), 3]
    pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
    return pred

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
