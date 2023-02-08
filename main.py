''' change working directory 
import os

os.chdir("C:/Users/zayn7lie/OneDrive - ber7/02-Prog/GitHub/ML-RMMD")
print('current working directory is {}'.format(os.getcwd()))

print(os.getcwd())
print(os.listdir(os.getcwd()))
'''

import numpy as np
# import ResNet

# normalize(preopera) figures
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
Transform = transforms.Compose([
    transforms.Resize(512), # short l -> 2048
    transforms.CenterCrop(512), # crop
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
train_dataset = datasets.ImageFolder(root="./OIA-ODIR/Training Set/Images", transform=Transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=8)

# import xlrd table to numpy.array
import xlrd
def rData(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [ [int(table.row_values(i,0,1)[0])] + table.row_values(i,-8) for i in range(1,table.nrows)]
    return np.array(data)
    #data: num, bool: NDGCAHMO

def main():
    data = rData("./OIA-ODIR/Off-site Test Set/Annotation/off-site test annotation (English).xlsx")
    d = data.flatten()
    print("finished")

if __name__ == "__main__":
    main()
