import numpy as np
import csv

import torch
from torchvision import transforms

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
class odirData(Dataset):
    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(512), # short l -> 512
            transforms.CenterCrop(512), # crop
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        with open(path + "/Annotation/label.csv") as label_o:
            # 0: file_name, 1-8: judge
            label_r = csv.reader(label_o)
            label_l = list(label_r)
            label = np.array(label_l)
        
        data_len = label.shape[0]
        self.filenames = []
        self.labels = []

        for i in range(data_len):
            self.filenames.append(label[i][0])
            self.labels.append([bool(l) for l in label[i][1:len(label)]])

    def __getitem__(self, index):
        img_name = self.path + "/Images/" + self.filenames[index]
        label = self.labels[index]
        
        img = plt.imread(img_name)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.filenames)
