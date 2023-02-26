from dataset import odirData
from train_eval import train, eval, adjust_lr
from model import Resnet50, RMMD

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler as SRS
from torch import optim
import os
from sklearn.model_selection import KFold # k-fold
import numpy as np
# import torch_xla.core.xla_model as xm #tputraining

def main():
    # para setting
    modellr = 1e-4
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    EPOCHS = 50 #50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # xm.xla_device() # 

    # load data
    fr_dataset = odirData("./OIA-ODIR/Off-site Test Set")
    to_dataset = odirData("./OIA-ODIR/On-site Test Set")
    fr_loader = DataLoader(fr_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    to_loader = DataLoader(to_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print("Origin:", len(fr_dataset), "=", len(fr_loader), "*", BATCH_SIZE, "->", len(to_dataset), "=", len(to_loader), "*", BATCH_SIZE, '\n')

    kfold = KFold(n_splits=10, shuffle=True)
    for fold_1, (fr_idx_9, fr_idx_1) in enumerate(kfold.split(np.arange(len(fr_dataset)))):
        for fold_2, (to_idx_9, to_idx_1) in enumerate(kfold.split(np.arange(len(to_dataset)))):
            
            # split data
            fr_tr_idxs, fr_ts_idxs = SRS(fr_idx_9), SRS(fr_idx_1)
            to_tr_idxs, to_ts_idxs = SRS(to_idx_9), SRS(to_idx_1)
            tr_loader_x = DataLoader(fr_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=fr_tr_idxs)
            tr_loader_y = DataLoader(to_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=to_tr_idxs)
            ts_loader_y = DataLoader(fr_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=fr_tr_idxs)
            ts_loader_y = DataLoader(to_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=to_ts_idxs)
            # print("K-fold:", fr_idx_9, "+", to_idx_9, "->", to_idx_1)
            print("K-fold:", len(tr_loader_x), "*", BATCH_SIZE, "+", len(tr_loader_y), "*", BATCH_SIZE,  "->", len(ts_loader_y), "*", BATCH_SIZE, '\n')
            
            # load model
            model = RMMD()
            model.to(DEVICE)
            if os.path.exists("./modelCache.pt"):
                model.load_state_dict(torch.load("./modelCache.pt"))

            # train model
            optimizer = optim.Adam(model.parameters(), lr=modellr)
            for epoch in range(1, EPOCHS + 1):
                adjust_lr(optimizer, epoch, modellr)
                train(epoch, model, DEVICE, tr_loader_x, tr_loader_y, optimizer)
            torch.save(model.state_dict(), "./modelCache.pt")
            
            # evaluate model
            eval(model, DEVICE, ts_loader_y)

            break
        break


    return 0

    
if __name__ == "__main__":
    main()
