#''' change working directory 
import os

os.chdir("C:/Users/zayn7lie/OneDrive - ber7/02-Prog/GitHub/ML-RMMD")
print('current working directory is {}'.format(os.getcwd()))

print(os.getcwd())
print(os.listdir(os.getcwd()))
#'''

from dataset import odirData
import model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
def main():
    # load data
    train_dataset = odirData("./OIA-ODIR/Training Set")
    test_dataset = odirData("./OIA-ODIR/Off-site Test Set")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    print(train_dataset.__getitem__(1))

    
if __name__ == "__main__":
    main()
