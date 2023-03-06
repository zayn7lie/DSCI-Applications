from dataset import odirData
from model import RMMD
from train_eval import train, eval, adjust_lr
from model import BCEFocalLosswithLogits

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler as SRS
from torch import optim
from sklearn.model_selection import KFold # k-fold
import numpy as np
import os
# import torch_xla.core.xla_model as xm #tputraining

def main():
    # para setting
    modellr = 1e-4
    BATCH_SIZE = 20
    NUM_WORKERS = 1
    EPOCHS = 60
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # xm.xla_device()
    K = 10 # k-fold
    criterion = BCEFocalLosswithLogits() # torch.nn.BCEWithLogitsLoss()
    ld = 0.0000 #lambda

    # load data
    fr_dataset = odirData("./OIA-ODIR/Off-site Test Set")
    to_dataset = odirData("./OIA-ODIR/On-site Test Set")

    # K-fold
    kfold = KFold(n_splits=K, shuffle=True)
    fr_idx = iter(enumerate(kfold.split(np.arange(len(fr_dataset)))))
    to_idx = iter(enumerate(kfold.split(np.arange(len(to_dataset)))))
    for i in range(K):
        print("## {:.0f}th FOLD:".format(i + 1))
        fold_1, (fr_idx_tr, fr_idx_ts) = fr_idx.__next__()
        fold_2, (to_idx_tr, to_idx_ts) = to_idx.__next__()

        # split data
        fr_tr_idxs, fr_ts_idxs = SRS(fr_idx_tr), SRS(fr_idx_ts)
        to_tr_idxs, to_ts_idxs = SRS(to_idx_tr), SRS(to_idx_ts)
        tr_loader_x = DataLoader(fr_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=fr_tr_idxs)
        tr_loader_y = DataLoader(to_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=to_tr_idxs)
        ts_loader_x = DataLoader(fr_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=fr_ts_idxs)
        ts_loader_y = DataLoader(to_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, sampler=to_ts_idxs)
        # print("K-fold:", fr_idx_9, "+", to_idx_9, "->", to_idx_1)
        for ld in [0.025, 0]:
            # load model
            print("\n### LAMBDA = {:.4f}\n".format(ld))
            model = RMMD()
            model.to(DEVICE)
            if os.path.exists("./modelCache.pt"):
                model.load_state_dict(torch.load("./modelCache.pt"))

            # train model
            optimizer = optim.Adam(model.parameters(), lr=modellr)
            for epoch in range(EPOCHS):
                adjust_lr(optimizer, epoch, modellr)
                train(epoch + 1, model, DEVICE, tr_loader_x, tr_loader_y, optimizer, criterion, ld)
            torch.save(model.state_dict(), "./modelCache_{:.0f}.pt".format(ld * 10000))
            
            # evaluate model
            eval(model, DEVICE, ts_loader_y)

        break
    return 0

    
if __name__ == "__main__":
    main()
