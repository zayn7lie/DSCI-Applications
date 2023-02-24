from dataset import odirData
import torch
from torch.utils.data import DataLoader
from torch import optim
from model import Resnet50
from train_eval import lrAdjust, train, eval
from tools import workDirChanger
import os

def main():
    workDirChanger("C:/Users/zayn7lie/OneDrive - ber7/02-Prog/GitHub/ML-RMMD")

    # para setting
    modellr = 1e-4
    BATCH_SIZE = 20
    EPOCHS = 30
    DEVICE = torch.device('cpu')#'cuda')

    # load data
    train_dataset = odirData("./OIA-ODIR/Training Set")
    test_dataset = odirData("./OIA-ODIR/Off-site Test Set")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    # model load or create
    model = Resnet50(len(train_dataset))
    model.to(DEVICE)
    if os.listdir("./modelCache"):
        model = model.load_state_dict(torch.load("./modelCache"))
    else: 
        optimizer = optim.Adam(model.parameters(), lr=modellr)
        for epoch in range(1, EPOCHS + 1):
            lrAdjust(modellr, optimizer, epoch)
            train(model, DEVICE, train_loader, optimizer, epoch)
        torch.save(model.state_dict(), "./modelCache")
        
    
    eval(model, DEVICE, test_loader)

    return 0

    
if __name__ == "__main__":
    main()
