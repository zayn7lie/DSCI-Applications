from dataset import odirData
import torch
from torch.utils.data import DataLoader
from torch import optim
from model import Resnet50
from train_eval import train, eval, adjust_lr
from tools import workDirChanger
import os

def main():
    # workDirChanger("C:/Users/zayn7lie/OneDrive - ber7/02-Prog/GitHub/ML-RMMD")

    # para setting
    modellr = 1e-4
    BATCH_SIZE = 72
    EPOCHS = 1 # 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    train_dataset = odirData("./OIA-ODIR/On-site Test Set")
    test_dataset = odirData("./OIA-ODIR/Off-site Test Set")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=1)
    print('Train_dataset size:', len(train_dataset), "=", len(train_loader), "*", BATCH_SIZE)
    print('Test_dataset size:', len(test_dataset), "=", len(test_loader), "*", BATCH_SIZE, '\n')


    # model load or create
    model = Resnet50(len(train_dataset))
    model.to(DEVICE)
    if 0: # os.listdir("./modelCache.zip"):
        model = model.load_state_dict(torch.load("./modelCache"))
    else: 
        optimizer = optim.Adam(model.parameters(), lr=modellr)
        for epoch in range(1, EPOCHS + 1):
            adjust_lr(optimizer, epoch, modellr)
            train(model, DEVICE, train_loader, optimizer, epoch)
        # torch.save(model.state_dict(), "/ML-RMMD")
        
    
    eval(model, DEVICE, test_loader)

    return 0

    
if __name__ == "__main__":
    main()
