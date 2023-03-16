from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import csv

from PIL import Image

class odirData(Dataset):
    def __init__(self, path, TF=False):
        self.path = path
        self.TF = TF
        self.transform_o = transforms.Compose([
            transforms.Resize(168),
            transforms.CenterCrop(224), # crop
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_f = transforms.Compose([
            transforms.Resize(168),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        with open(path + "/Annotation/label.csv", 'r', encoding = 'utf-8-sig') as label_o:
            # 0: file_name, 1-8: judge
            label_r = csv.reader(label_o)
            label_l = list(label_r)
            label = np.array(label_l)
        
        self.filenames = []
        self.labels = []
        self.img_folder = self.path + "/Images"
        for i in range(label.shape[0]):
            self.filenames.append(label[i][0])
            self.labels.append([float(label[i][1]), float(label[i][2]), float(label[i][3]), float(label[i][4]), float(label[i][5]), float(label[i][6]), float(label[i][7]), float(label[i][8])])
            # 1N 2D 3G 4C 5A 6H 7M 8O

    def __getitem__(self, index):
        img_name = self.img_folder + "/" + self.filenames[index]
        label = []
        # print(self.filenames[index])
        # print(self.labels[index])
        # for i in range(0, 8):
            # label.append(tensor(self.labels[index][i])) # self.labels[index][i]
        label = np.array(self.labels[index], dtype=float)

        # img = plt.imread(img_name)
        img = Image.open(img_name)
        if self.TF:
            img = self.transform_f(img)
        else:
            img = self.transform_o(img)
        # print(label)
        return img, label

    def __len__(self):
        return len(self.filenames)
