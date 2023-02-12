import xlrd
import numpy as np

def rxl(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [ [int(table.row_values(i,0,1)[0])] + table.row_values(i,-8) for i in range(1,table.nrows)]
    return np.array(data)
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
        lab = rxl(path_lab)
