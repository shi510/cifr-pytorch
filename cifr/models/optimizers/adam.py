import torch

from ..builder import OPTIMIZERS

OPTIMIZERS.register_module(name='Adam', module=torch.optim.Adam)
