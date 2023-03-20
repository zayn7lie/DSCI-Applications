from dataset import odirData
from model import RMMD
from net_train import train, adjust_lr
from net_eval import eval
from model import BCELogitsFocalLoss

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler as SRS
from torch import optim
from sklearn.model_selection import KFold # k-fold
import numpy as np
import os

# para setting
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # xm.xla_device()
K = 10 # k-fold

BATCH_SIZE = 15
NUM_WORKERS = 2

modellr = 1e-4
times = 100
param = 0.1
EPOCHS = 60

compare = [1e-6] #lambda
criterion = BCELogitsFocalLoss() # torch.nn.BCEWithLogitsLoss() # 
TF = True
DropBlock = True

fr_dataset = odirData("./OIA-ODIR/Training Set", TF=TF)
to_dataset = odirData("./OIA-ODIR/Off-site Test Set", TF=TF)

def main():
    # K-fold
    kfold = KFold(n_splits=K, shuffle=True)
    fr_idx = iter(enumerate(kfold.split(np.arange(len(fr_dataset)))))
    to_idx = iter(enumerate(kfold.split(np.arange(len(to_dataset)))))
    for i in range(K):
        print("## {:.0f}th FOLD:\n".format(i + 1))
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
        for ld in compare:
            # load model
            epochs = EPOCHS
            print("### LAMBDA = {:.0f} * 1e-7\n".format(ld * 1e7))
            model = RMMD(DropBlock=DropBlock)
            if os.path.exists("./modelCache.pt"):
                model.load_state_dict(torch.load("./modelCache.pt"))
                print("Model Loaded\n")

            # train model
            model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=modellr)
            for epoch in range(epochs):
                adjust_lr(optimizer, epoch, modellr, times=times, param=param)
                train(epoch + 1, model, DEVICE, tr_loader_x, tr_loader_y, optimizer, criterion, ld=ld, BATCH_SIZE=BATCH_SIZE)
            
            torch.save(model.state_dict(), "./modelCache_{:.0f}.pt".format(ld * 1e6))
            
            # evaluate model
            eval(model, DEVICE, ts_loader_y)

        break
    return 0

    
if __name__ == "__main__":
    main()
