from dataset import odirData
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from model import Resnet50
from train_eval import train, eval, adjust_lr
from tools import workDirChanger
import os
from sklearn.model_selection import KFold # k-fold
import numpy as np
# import torch_xla.core.xla_model as xm #tputraining

def main():
    # workDirChanger("C:/Users/zayn7lie/OneDrive - ber7/02-Prog/GitHub/ML-RMMD")

    # para setting
    modellr = 1e-4
    BATCH_SIZE = 50
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # xm.xla_device() # 

    # load data
    fr_dataset = odirData("./OIA-ODIR/On-site Test Set")
    to_dataset = odirData("./OIA-ODIR/Off-site Test Set")
    fr_loader = DataLoader(fr_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
    to_loader = DataLoader(to_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
    print("Origin:", len(fr_dataset), "=", len(fr_loader), "*", BATCH_SIZE, "->", len(to_dataset), "=", len(to_loader), "*", BATCH_SIZE, '\n')


    # model load or create
    kfold = KFold(n_splits=10, shuffle=True)
    for fold_1, (fr_idx_9, fr_idx_1) in enumerate(kfold.split(np.arange(len(fr_dataset)))):
        for fold_2, (to_idx_9, to_idx_1) in enumerate(kfold.split(np.arange(len(to_dataset)))):
            fr_tr_idxs = SubsetRandomSampler(fr_idx_9)
            fr_ts_idxs = SubsetRandomSampler(fr_idx_1)
            to_tr_idxs = SubsetRandomSampler(to_idx_9)
            to_ts_idxs = SubsetRandomSampler(to_idx_1)

            # tr_dataset = TensorDataset(fr_tr_idxs, to_tr_idxs)
            # ts_dataset = TensorDataset(to_ts_idxs, fr_ts_idxs)
            tr_loader = DataLoader(fr_dataset, BATCH_SIZE, num_workers=2, sampler=fr_tr_idxs)
            ts_loader = DataLoader(to_dataset, BATCH_SIZE, num_workers=2, sampler=to_ts_idxs)
            print("K-fold:", fr_idx_9, "->", to_idx_1)
            print("K-fold:", len(tr_loader), "*", BATCH_SIZE, "->", len(to_loader), "*", BATCH_SIZE, '\n')
            model = Resnet50(len(tr_loader))
            model.to(DEVICE)
            if os.path.exists("./modelCache.pt"):
                model = model.load_state_dict(torch.load("./modelCache.pt"))
            else: 
                optimizer = optim.Adam(model.parameters(), lr=modellr)
                for epoch in range(1, EPOCHS + 1):
                    adjust_lr(optimizer, epoch, modellr)
                    train(epoch, model, DEVICE, tr_loader, optimizer)
                torch.save(model.state_dict(), "./modelCache.pt")
                

            eval(model, DEVICE, ts_loader)
            break
        break


    return 0

    
if __name__ == "__main__":
    main()
