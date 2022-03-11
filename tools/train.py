import argparse
import os
import math

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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

def batched_predict(model, coord, cell, bsize):
    n = coord.shape[1]
    ql = 0
    preds = []
    while ql < n:
        qr = min(ql + bsize, n)
        pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :]).detach()
        
        preds.append(pred)
        ql = qr
    pred = torch.cat(preds, dim=1)
    return pred

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def train(args, config):
    model = build_architecture(config.model).cuda()
    disc = build_discriminator(config.discriminator).cuda()
    config.optimizer.update({'params': model.parameters()})
    optim_m = build_optimizer(config.optimizer)
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
            model.train()
            disc.train()

            #
            # Discriminator Step
            #
            requires_grad(model, False)
            requires_grad(disc, True)
            optim_d.zero_grad()

            lr = batch_gan['lr'].cuda()
            coord = batch_gan['coord'].cuda()
            cell = batch_gan['cell'].cuda()
            real = batch_gan['real'].cuda()

            model.gen_feat(lr)
            fake = batched_predict(model, coord, cell, 1024)
            fake.clamp_(0, 1)
            ih, iw = lr.shape[-2:]
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [lr.shape[0], round(ih * s), round(iw * s), 3]
            fake = fake.view(*shape).permute(0, 3, 1, 2).contiguous()

            fake_pred = grad_norm_fn(disc, fake).detach()
            real_pred = grad_norm_fn(disc, real)

            rel_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
            loss_rel_real_d = rel_loss_fn(real_pred - torch.mean(fake_pred), torch.ones_like(real_pred)).mean()
            loss_rel_fake_d = rel_loss_fn(fake_pred - torch.mean(real_pred.detach()), torch.zeros_like(fake_pred)).mean()
            loss_rel_d = (loss_rel_real_d + loss_rel_fake_d) / 2
            loss_d = loss_rel_d
            loss_d.backward()
            optim_d.step()

            #
            # Generator Step
            #
            requires_grad(model, True)
            # requires_grad(disc, False)
            optim_m.zero_grad()

            model.gen_feat(lr)
            fake = batched_predict(model, coord, cell, 1024)
            fake.clamp_(0, 1)
            ih, iw = lr.shape[-2:]
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [lr.shape[0], round(ih * s), round(iw * s), 3]
            fake = fake.view(*shape).permute(0, 3, 1, 2).contiguous()
            fake_pred = grad_norm_fn(disc, fake).detach()
            real_pred = grad_norm_fn(disc, real)
            loss_g_l1 = torch.nn.functional.l1_loss(fake, real, reduction='mean') * 1e-2
            rel_loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
            loss_rel_real = rel_loss_fn(real_pred - torch.mean(fake_pred), torch.zeros_like(real_pred)).mean()
            loss_rel_fake = rel_loss_fn(fake_pred - torch.mean(real_pred.detach()), torch.ones_like(fake_pred)).mean()
            loss_rel_g = (loss_rel_real + loss_rel_fake) / 2
            ctx_loss = contextual_loss(fake, real)

            inp = batch['inp'].cuda()
            coord = batch['coord'].cuda()
            cell = batch['cell'].cuda()
            gt = batch['gt'].cuda()

            pred = model(inp, coord, cell)
            hr_l1_loss = loss_fn(pred, gt)

            loss_g = loss_g_l1 + loss_rel_g + hr_l1_loss + ctx_loss

            loss_g.backward()
            optim_m.step()

            loss_str = f'd_loss: {loss_d:.4f}; g_loss: {loss_g:.4f}; hr_l1_loss: {hr_l1_loss:.4f}'
            iter_pbar.set_description(loss_str)

        torch.save(
            model.state_dict(),
            f'{WORK_DIR}/{config_name}/checkpoints/{epoch+1:0>6}.pt')
        model.eval()
        with torch.no_grad():
            it = iter(test_loader)
            for i in range(20):
                test_batch = it.next()
                inp = test_batch['inp'].cuda()
                coord = test_batch['coord'].cuda()
                cell = test_batch['cell'].cuda()
                gt = test_batch['gt'].cuda()
                model.gen_feat(inp)
                pred = batched_predict(model, coord, cell, 1024)
                pred.clamp_(0, 1)
                ih, iw = inp.shape[-2:]
                s = math.sqrt(coord.shape[1] / (ih * iw))
                shape = [inp.shape[0], round(ih * s), round(iw * s), 3]
                pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
                gt = gt.view(*shape).permute(0, 3, 1, 2).contiguous()
                add_img_plot(fig, inp[0], f'Input', rows, cols, i*3+1)
                add_img_plot(fig, pred[0], f'Predict', rows, cols, i*3+2)
                add_img_plot(fig, gt[0], f'GT', rows, cols, i*3+3)
        plt.tight_layout()
        plt.savefig(f'{WORK_DIR}/{config_name}/images/train_{count:0>6}.jpg', bbox_inches='tight')
        plt.clf()
        count += 1
        iter_pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    train(args, cfg)
