import argparse
import os

import torch
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch import optim
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from cifr.core.config import Config
from cifr.models.builder import build_architecture, build_optimizer, build_dataset
from cifr.models.builder import build_discriminator
from cifr.models.losses.contextual_loss import ContextualLoss, ContextualBilateralLoss
from cifr.models.losses.gradient_norm import normalize_gradient
from cifr.models.losses.gan_loss import d_logistic_loss
from cifr.models.losses.gan_loss import g_nonsaturating_loss
from tools.utils import query_all_pixels
from tools.utils import requires_grad
from tools.utils import save_pred_img


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

    scheduler_g = optim.lr_scheduler.StepLR(optim_g, step_size=50, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optim_d, step_size=50, gamma=0.5)

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
        scheduler_g.step()
        scheduler_d.step()
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
