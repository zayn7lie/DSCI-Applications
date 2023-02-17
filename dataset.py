import numpy as np
import pandas as pd
# data: num, bool: NDGCAHMO

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class getdata(Dataset):
    def __init__(self, path_img, path_lab):
        self.transform = transforms.Compose([
            transforms.Resize(512), # short l -> 512
            transforms.CenterCrop(512), # crop
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
