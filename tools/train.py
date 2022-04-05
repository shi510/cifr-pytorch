import argparse
import os
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from cifr.core.config import Config
from cifr.models.builder import build_generator, build_optimizer, build_dataset
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

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def save_pred_img(generator, data_loader, img_path, fig, rows, cols):
    it = iter(data_loader)
    for i in range(20):
        test_batch = it.next()
        inp = test_batch['inp'].cuda()
        coord = test_batch['coord'].cuda()
        cell = test_batch['cell'].cuda()
        gt = test_batch['gt'].cuda()
        pred = generator(inp, coord, cell)
        gt = gt.view([pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[1]])
        gt = gt.permute(0, 3, 1, 2).contiguous()
        add_img_plot(fig, inp[0], f'Input', rows, cols, i*3+1)
        add_img_plot(fig, pred[0], f'Predict', rows, cols, i*3+2)
        add_img_plot(fig, gt[0], f'GT', rows, cols, i*3+3)
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight')
    plt.clf()

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def train(args, config):
    gen = build_generator(config.generator).cuda()
    gen_ema = ExponentialMovingAverage(gen.parameters(), decay=0.995)
    disc = build_discriminator(config.discriminator).cuda()

    config.optimizer.update({'params': gen.parameters()})
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
    rows = 20
    cols = 3
    fig = plt.figure(figsize=(15, rows*6))

    total_iter = len(train_set) // config.batch_size + int(len(train_set) % config.batch_size > 0)
    epoch_pbar = tqdm(range(config.epoch), total=config.epoch, desc='Epoch', position=0, ncols=0)
    for epoch in epoch_pbar:
        iter_pbar = tqdm(enumerate(zip(train_loader, train_loader_gan)), total=total_iter, leave=False, position=1, ncols=0)
        for n, (batch, batch_gan) in iter_pbar:
            gen.train()
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

            fake = gen(lr, coord, cell)
            fake_pred = grad_norm_fn(disc, fake)
            ctx_loss = contextual_loss(fake, real)
            loss_fake = g_nonsaturating_loss(fake_pred)
            loss_g = ctx_loss + loss_fake
            loss_g.backward()

            query_pred = gen.query_rgb(batch['inp'].cuda(), batch['coord'].cuda(), batch['cell'].cuda())
            query_l1_loss = loss_fn(query_pred, batch['gt'].cuda())
            query_l1_loss.backward()

            optim_g.step()
            gen_ema.update()

            #
            # Discriminator Step
            #
            requires_grad(disc, True)
            optim_d.zero_grad()

            fake_pred = grad_norm_fn(disc, fake.detach())
            real_pred = grad_norm_fn(disc, real)
            loss_d = d_logistic_loss(real_pred, fake_pred)
            loss_d.backward()
            optim_d.step()

            loss_str = f'd: {loss_d:.4f};'
            loss_str += f' g: {loss_g:.4f};'
            loss_str += f' g_ctx: {ctx_loss:.4f}'
            loss_str += f' query_l1: {query_l1_loss:.4f}'
            iter_pbar.set_description(loss_str)

        torch.save(
            {
                'generator': gen.state_dict(),
                'generator_ema': gen_ema.state_dict(),
                'discriminator': disc.state_dict(),
            },
            f'{WORK_DIR}/{config_name}/checkpoints/{epoch+1:0>6}.pth'
        )
        gen_ema.store(gen.parameters())
        gen_ema.copy_to(gen.parameters())
        gen.eval()
        img_path = f'{WORK_DIR}/{config_name}/images/train_{epoch+1:0>6}.jpg'
        save_pred_img(gen, test_loader, img_path, fig, rows, cols)
        gen_ema.restore(gen.parameters())
        iter_pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    train(args, cfg)
