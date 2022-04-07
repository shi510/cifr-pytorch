import math

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader

def add_img_plot(
    fig:plt.Figure,
    img:torch.Tensor,
    title:str,
    rows:int,
    cols:int,
    num:int
):
    img = img * 0.5 + 0.5
    img.clamp_(0, 1)
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = img.transpose((1, 2, 0))
    ax = fig.add_subplot(rows, cols, num)
    ax.imshow(img)
    ax.set_title(title, size=25)
    ax.axis("off")

def requires_grad(model:torch.nn.Module, flag:bool=True):
    for p in model.parameters():
        p.requires_grad = flag

def query_all_pixels(
    encoder:torch.nn.Module,
    model:torch.nn.Module,
    lr:torch.Tensor,
    coord:torch.Tensor,
    cell:torch.Tensor,
    bsize:int
):
    feature = encoder(lr)
    split_size = coord.shape[1] // bsize
    coord_list = torch.tensor_split(coord, split_size, 1)
    cell_list = torch.tensor_split(cell, split_size, 1)
    preds = []
    for coord_batch, cell_batch in zip(coord_list, cell_list):
        pred = model(lr, feature, coord_batch, cell_batch)
        preds.append(pred)
    pred = torch.cat(preds, dim=1)
    ih, iw = lr.shape[-2:]
    s = math.sqrt(coord.shape[1] / (ih * iw))
    shape = [lr.shape[0], round(ih * s), round(iw * s), 3]
    pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
    return pred

@torch.no_grad()
def save_pred_img(
    encoder:torch.nn.Module,
    model:torch.nn.Module,
    data_loader:DataLoader,
    img_path:str,
    fig:plt.Figure,
    rows:int,
    cols:int
):
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
