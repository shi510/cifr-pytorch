import math

import torch
import torch.nn as nn

from ..builder import GENERATORS
from ..builder import build_architecture


@GENERATORS.register_module()
class LIIFGenerator(nn.Module):
    def __init__(self,
        encoder_model,
        implicit_model,
        query_batch=1024,
    ):
        super(LIIFGenerator, self).__init__()
        self.encoder_model = build_architecture(encoder_model)
        self.implicit_model = build_architecture(implicit_model)
        self.query_batch = query_batch

    def forward(self, inp, coord, cell):
        feature = self.encoder_model(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + self.query_batch, n)
            pred = self.implicit_model(inp, feature, coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
        pred.clamp_(0, 1)
        ih, iw = inp.shape[-2:]
        s = math.sqrt(coord.shape[1] / (ih * iw))
        shape = [inp.shape[0], round(ih * s), round(iw * s), 3]
        pred = pred.view(*shape).permute(0, 3, 1, 2).contiguous()
        return pred

    def query_rgb(self, inp, coord, cell):
        feature = self.encoder_model(inp)
        pred = self.implicit_model(inp, feature, coord, cell)
        return pred
