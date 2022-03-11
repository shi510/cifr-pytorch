import os

import torch
import numpy as np

from PIL import Image

from ..models.builder import DATASETS


@DATASETS.register_module()
class DIV2K(torch.utils.data.Dataset):
    def __init__(self, root, n_repeat=1):
        self.root = root
        self.n_repeat = n_repeat
        self.file_list = []
        for file_name in sorted(os.listdir(self.root)):
            file_path = os.path.join(self.root, file_name)
            if 'png' not in os.path.splitext(file_name)[1]:
                continue
            self.file_list.append(file_path)

    def __getitem__(self, index):
        img_path = self.file_list[index % len(self.file_list)]
        img = Image.open(img_path)
        return img.convert("RGB")

    def __len__(self):
        return len(self.file_list) * self.n_repeat
