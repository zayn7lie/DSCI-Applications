from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import csv

from PIL import Image

class odirData(Dataset):
    def __init__(self, path):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(224), # short l -> 512
            transforms.CenterCrop(224), # crop
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
            self.labels.append([label[i][1], label[i][2], label[i][3], label[i][4], label[i][5], label[i][6], label[i][7], label[i][8],])
            # 1N 2D 3G 4C 5A 6H 7M 8O

    def __getitem__(self, index):
        img_name = self.img_folder + "/" + self.filenames[index]
        label = self.labels[index]
        
        # img = plt.imread(img_name)
        img = Image.open(img_name).convert('RGB')
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.filenames)
