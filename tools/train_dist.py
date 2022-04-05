import argparse
import os
import math
from re import M

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from cifr.core.config import Config
from cifr.models.builder import build_architecture, build_optimizer, build_dataset
from cifr.models.builder import build_discriminator
from cifr.models.losses.contextual_loss import ContextualLoss, ContextualBilateralLoss
from cifr.models.losses.gradient_norm import normalize_gradient


WORK_DIR = './work_dir'

def synchronize():
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

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

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def train(args, config, device):
    model = build_architecture(config.model).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    encoder = build_architecture(config.encoder).to(device)
    encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)

    disc = build_discriminator(config.discriminator).to(device)
    disc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(disc)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist.get_rank()],
        output_device=dist.get_rank(),
        broadcast_buffers=False
    )
    encoder = torch.nn.parallel.DistributedDataParallel(
        encoder,
        device_ids=[dist.get_rank()],
        output_device=dist.get_rank(),
        broadcast_buffers=False
    )
    disc = torch.nn.parallel.DistributedDataParallel(
        disc,
        device_ids=[dist.get_rank()],
        output_device=dist.get_rank(),
        broadcast_buffers=False
    )

    config.optimizer.update({'params': [
        {'params': encoder.parameters()},
        {'params': model.parameters()}
        ]})
    optim_g = build_optimizer(config.optimizer)
    config.optimizer.update({'params': disc.parameters()})
    optim_d = build_optimizer(config.optimizer)

    model_ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    encoder_ema = ExponentialMovingAverage(encoder.parameters(), decay=0.995)

    train_set_gan = build_dataset(config.train_dataset_gan)
    train_set = build_dataset(config.train_dataset)
    test_set = build_dataset(config.test_dataset)

    train_loader_gan = torch.utils.data.DataLoader(
        train_set_gan,
        batch_size=config.batch_size,
        num_workers=6,
        drop_last=True,
        sampler=DistributedSampler(train_set_gan, shuffle=True),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=6,
        drop_last=True,
        sampler=DistributedSampler(train_set, shuffle=True),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        num_workers=1
    )
    contextual_loss = ContextualLoss(use_vgg=True, vgg_layer="conv5_4").to(device)
    loss_fn = torch.nn.L1Loss()
    grad_norm_fn = normalize_gradient if config.discriminator_gradient_norm else lambda fn, x: fn(x)

    config_name = os.path.splitext(os.path.basename(args.config))[0] if args.name is None else args.name
    os.makedirs(f'{WORK_DIR}/{config_name}/images', exist_ok=True)
    os.makedirs(f'{WORK_DIR}/{config_name}/checkpoints', exist_ok=True)
    # config.dump(f'{WORK_DIR}/{config_name}/{config_name}.py')
    rows = 20
    cols = 3
    fig = plt.figure(figsize=(15, rows*6))

    total_iter = len(train_set) // config.batch_size // dist.get_world_size()
    epoch_pbar = tqdm(
        range(config.epoch),
        total=config.epoch,
        desc='Epoch',
        position=0,
        ncols=0,
        disable=dist.get_rank()!=0
    )
    for epoch in epoch_pbar:
        iter_pbar = tqdm(
            enumerate(zip(train_loader, train_loader_gan)),
            total=total_iter,
            leave=False,
            position=1,
            ncols=0,
            disable=dist.get_rank()!=0
        )
        for n, (batch, batch_gan) in iter_pbar:
            encoder.train()
            model.train()
            disc.train()

            lr = batch_gan['lr'].to(device)
            coord = batch_gan['coord'].to(device)
            cell = batch_gan['cell'].to(device)
            real = batch_gan['real'].to(device)

            #
            # Generator Step
            #
            requires_grad(disc, False)
            optim_g.zero_grad()

            fake = query_all_pixels(encoder, model, lr, coord, cell, 1024)
            fake_pred = grad_norm_fn(disc, fake)
            ctx_loss = contextual_loss(fake, real)
            loss_fake = g_nonsaturating_loss(fake_pred)
            loss_g = ctx_loss + loss_fake
            loss_g.backward()

            query_inp = batch['inp'].to(device)
            query_coord = batch['coord'].to(device)
            query_cell = batch['cell'].to(device)
            query_gt = batch['gt'].to(device)
            feature = encoder(query_inp)
            query_pred = model(query_inp, feature, query_coord, query_cell)
            query_l1_loss = loss_fn(query_pred, query_gt)
            query_l1_loss.backward()

            optim_g.step()
            encoder_ema.update()
            model_ema.update()

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

            loss_dict = {
                'd': loss_d,
                'g': loss_g,
                'g_ctx': ctx_loss,
                'query_l1': query_l1_loss
            }
            reduced_loss = reduce_loss_dict(loss_dict)
            if dist.get_rank() == 0:
                loss_d = reduced_loss['d']
                loss_g = reduced_loss['g']
                ctx_loss = reduced_loss['g_ctx']
                query_l1_loss = reduced_loss['query_l1']
                loss_str = f'd: {loss_d:.4f};'
                loss_str += f' g: {loss_g:.4f};'
                loss_str += f' g_ctx: {ctx_loss:.4f}'
                loss_str += f' query_l1: {query_l1_loss:.4f}'
                iter_pbar.set_description(loss_str)
        if dist.get_rank() == 0:
            torch.save(
                {
                    'encoder': encoder.module.state_dict(),
                    'model': model.module.state_dict(),
                    'encoder_ema': encoder_ema.state_dict(),
                    'model_ema': model_ema.state_dict(),
                    'discriminator': disc.module.state_dict(),
                },
                f'{WORK_DIR}/{config_name}/checkpoints/{epoch+1:0>6}.pth'
            )
            encoder_ema.store(encoder.parameters())
            model_ema.store(model.parameters())
            encoder_ema.copy_to(encoder.parameters())
            model_ema.copy_to(model.parameters())
            encoder.eval()
            model.eval()
            img_path = f'{WORK_DIR}/{config_name}/images/train_{epoch+1:0>6}.jpg'
            save_pred_img(encoder, model, test_loader, img_path, fig, rows, cols)
            encoder_ema.restore(encoder.parameters())
            model_ema.restore(model.parameters())
            iter_pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f'cuda:{local_rank}')
    synchronize()

    cfg = Config.fromfile(args.config)
    train(args, cfg, device)
