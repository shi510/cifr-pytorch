import argparse
import os
import math

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from cifr.core.config import Config
from cifr.models.builder import build_architecture, build_optimizer, build_dataset
from cifr.models.builder import build_discriminator
from cifr.models.losses.contextual_loss import ContextualLoss, ContextualBilateralLoss
from cifr.models.losses.gradient_norm import normalize_gradient


WORK_DIR = './work_dir'

def add_img_plot(fig, img, title, rows, cols, num):
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose((1, 2, 0))
    ax = fig.add_subplot(rows, cols, num)
    ax.imshow(img)
    ax.set_title(title, size=25)
    ax.axis("off")

def query_all_pixels(encoder, model, lr, coord, cell, bsize):
    feature = encoder(lr)
    n = coord.shape[1]
    ql = 0
    preds = []
    while ql < n:
        qr = min(ql + bsize, n)
        pred = model(feature, coord[:, ql: qr, :], cell[:, ql: qr, :])
        preds.append(pred)
        ql = qr
    pred = torch.cat(preds, dim=1)
    pred.clamp_(0, 1)
    ih, iw = lr.shape[-2:]
    s = math.sqrt(coord.shape[1] / (ih * iw))
    shape = [lr.shape[0], round(ih * s), round(iw * s), 3]
    pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
    return pred

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def save_pred_img(encoder, model, data_loader, img_path, fig, rows, cols):
    it = iter(data_loader)
    for i in range(20):
        test_batch = it.next()
        inp = test_batch['inp'].cuda()
        coord = test_batch['coord'].cuda()
        cell = test_batch['cell'].cuda()
        gt = test_batch['gt'].cuda()
        pred = query_all_pixels(encoder, model, inp, coord, cell, 1024)
        gt = gt.view([pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[1]])
        gt = gt.permute(0, 3, 1, 2).contiguous()
        add_img_plot(fig, inp[0], f'Input', rows, cols, i*3+1)
        add_img_plot(fig, pred[0], f'Predict', rows, cols, i*3+2)
        add_img_plot(fig, gt[0], f'GT', rows, cols, i*3+3)
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    plt.clf()

def train(args, config):
    model = build_architecture(config.model).cuda()
    encoder = build_architecture(config.encoder).cuda()
    model_ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    encoder_ema = ExponentialMovingAverage(encoder.parameters(), decay=0.995)
    disc = build_discriminator(config.discriminator).cuda()

    config.optimizer.update({'params': [
        {'params': encoder.parameters()},
        {'params': model.parameters()}
        ]})
    optim_g = build_optimizer(config.optimizer)
    config.optimizer.update({'params': disc.parameters()})
    optim_d = build_optimizer(config.optimizer)

    train_set_gan = build_dataset(config.train_dataset_gan)
    train_set = build_dataset(config.train_dataset)
    test_set = build_dataset(config.test_dataset)

    train_loader_gan = torch.utils.data.DataLoader(
        train_set_gan,
        batch_size=config.batch_size,
        num_workers=6,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=6,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        num_workers=1
    )
    contextual_loss = ContextualLoss(use_vgg=True, vgg_layer="conv5_4").cuda()
    loss_fn = torch.nn.L1Loss()
    grad_norm_fn = normalize_gradient if config.discriminator_gradient_norm else lambda fn, x: fn(x)

    if args.name == None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
    else:
        config_name = args.name
    os.makedirs(f'{WORK_DIR}/{config_name}/images', exist_ok=True)
    os.makedirs(f'{WORK_DIR}/{config_name}/checkpoints', exist_ok=True)
    # config.dump(f'{WORK_DIR}/{config_name}/{config_name}.py')
    count = 0
    rows = 20
    cols = 3
    fig = plt.figure(figsize=(15, rows*6))
    total_iter = len(train_set) // config.batch_size + int(len(train_set) % config.batch_size > 0)
    epoch_pbar = tqdm(range(config.epoch), total=config.epoch, desc='Epoch', position=0)
    for epoch in epoch_pbar:
        iter_pbar = tqdm(enumerate(zip(train_loader, train_loader_gan)), total=total_iter, leave=False, position=1)
        for n, (batch, batch_gan) in iter_pbar:
            encoder.train()
            model.train()
            disc.train()

            lr = batch_gan['lr'].cuda()
            coord = batch_gan['coord'].cuda()
            cell = batch_gan['cell'].cuda()
            real = batch_gan['real'].cuda()

            #
            # Generator Step
            #
            requires_grad(disc, False)
            optim_g.zero_grad()
            feature = encoder(batch['inp'].cuda())
            query_pred = model(feature, batch['coord'].cuda(), batch['cell'].cuda())
            query_l1_loss = loss_fn(query_pred, batch['gt'].cuda())
            query_l1_loss.backward()

            fake = query_all_pixels(encoder, model, lr, coord, cell, 1024)
            fake_pred = grad_norm_fn(disc, fake)
            real_pred = grad_norm_fn(disc, real).detach()
            loss_g_l1 = torch.nn.functional.l1_loss(fake, real, reduction='mean') * 1e-2
            rel_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
            loss_rel_real = rel_loss_fn(real_pred - torch.mean(fake_pred), torch.zeros_like(real_pred)).mean()
            loss_rel_fake = rel_loss_fn(fake_pred - torch.mean(real_pred), torch.ones_like(fake_pred)).mean()
            loss_rel_g = (loss_rel_real + loss_rel_fake) / 2
            ctx_loss = contextual_loss(fake, real)

            loss_g = loss_g_l1 + loss_rel_g + ctx_loss
            loss_g.backward()
            optim_g.step()
            encoder_ema.update()
            model_ema.update()

            #
            # Discriminator Step
            #
            requires_grad(disc, True)
            optim_d.zero_grad()

            rel_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
            fake_pred = grad_norm_fn(disc, fake).detach()
            real_pred = grad_norm_fn(disc, real)
            loss_rel_real_d = rel_loss_fn(real_pred - torch.mean(fake_pred), torch.ones_like(real_pred)).mean() * 0.5
            loss_rel_real_d.backward()

            fake_pred = grad_norm_fn(disc, fake.detach())
            loss_rel_fake_d = rel_loss_fn(fake_pred - torch.mean(real_pred.detach()), torch.zeros_like(fake_pred)).mean() * 0.5
            loss_rel_fake_d.backward()
            optim_d.step()
            loss_d = loss_rel_real_d + loss_rel_fake_d

            loss_str = f'd: {loss_d:.4f};'
            loss_str += f' g: {loss_g:.4f};'
            loss_str += f' g_l1: {loss_g_l1:.4f};'
            loss_str += f' g_rel: {loss_rel_g:.4f}'
            loss_str += f' g_ctx: {ctx_loss:.4f}'
            loss_str += f' query_l1: {query_l1_loss:.4f}'
            iter_pbar.set_description(loss_str)

        torch.save(
            {
                'encoder': encoder.state_dict(),
                'model': model.state_dict(),
                'encoder_ema': encoder_ema.state_dict(),
                'model_ema': model_ema.state_dict(),
                'discriminator': disc.state_dict(),
            },
            f'{WORK_DIR}/{config_name}/checkpoints/{epoch+1:0>6}.pth'
        )
        encoder_ema.store(encoder.parameters())
        model_ema.store(model.parameters())
        encoder_ema.copy_to(encoder.parameters())
        model_ema.copy_to(model.parameters())
        encoder.eval()
        model.eval()
        img_path = f'{WORK_DIR}/{config_name}/images/train_{count:0>6}.jpg'
        save_pred_img(encoder, model, test_loader, img_path, fig, rows, cols)
        encoder_ema.restore(encoder.parameters())
        model_ema.restore(model.parameters())
        count += 1
        iter_pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    train(args, cfg)
