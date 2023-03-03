import torch
from torch import nn
from model import FC

def adjust_lr(optimizer, epoch, modellr):
    modellrnew = modellr * (0.1 ** (epoch // 25)) # 25
    print("Epoch:", epoch + 1, "Learning Rate:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def train(epoch, model, device, tr_loader_x, tr_loader_y, optimizer, ld):
    criterion = FC() #nn.BCELoss() #
    model.train()
    sum_loss, sum_mmd, sum_bce = 0, 0, 0
    cnt = 0
    sumloss, minloss, maxloss = 0, 100, 0
    iter_y = iter(tr_loader_y)
    
    for batch_idx, (imgs_x, targets) in enumerate(tr_loader_x):
        imgs_x, targets = imgs_x.to(device), targets.to(device)
        
        imgs_y, _ = iter_y.__next__()
        imgs_y = imgs_y.to(device)
        cnt += 1
        if cnt % len(tr_loader_y) == 0:
            iter_y = iter(tr_loader_y)

        optimizer.zero_grad()

        output, mmd_loss = model(imgs_x, imgs_y)
        bce_loss = criterion(output, targets.type(torch.float))
        loss = bce_loss + ld * mmd_loss
        
        sumloss += loss.item()
        minloss, maxloss = min(minloss, loss), max(maxloss, loss)
        
        sum_loss += loss.item()
        sum_mmd += mmd_loss.item()
        sum_bce += bce_loss.item()

        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 15 == 0:
            print("- [{:.0f}/{:.0f}] Loss: AVG={:.6f} MAX={:.6f} MIN={:.6f}".format((batch_idx + 1), len(tr_loader_x), sumloss / 45, maxloss, minloss))
            sumloss, minloss, maxloss = 0, 100, 0

    avg_loss, avg_mmd, avg_bce = sum_loss * 100 / len(tr_loader_x), sum_mmd * 100 / len(tr_loader_x), sum_bce * 100 / len(tr_loader_x)
    print("Epoch: {:.0f} Loss: {:.11f}% MMD: {:.11f}% BCE: {:.11f}%\n".format(epoch, avg_loss, avg_mmd, avg_bce))

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

def eval(model, device, test_loader):
    criterion = FC()
    model.eval()
    test_loss, cnt = 0, 0
    f1, auc = 0, 0
    # total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            output, _ = model(imgs, None)
            loss = criterion(output, targets.type(torch.float))
            
            print_loss = loss.item()
            test_loss += print_loss

            output = output.cpu().detach().numpy().flat
            targets = targets.cpu().detach().numpy().flat

            output = np.array(output >= 0.5, dtype=float)
            # print(output)
            
            cnt += 1
            f1 += f1_score(targets, output)
            auc += roc_auc_score(targets, output)

        avgloss = test_loss / cnt
        avgf1 = f1 * 100 / cnt
        avgauc = auc * 100 / cnt
        print('\nVal set: BCE: {:.4f} Average f1: {:.4f}% Average auc: {:.4f}%\n'.format(avgloss, avgf1, avgauc))
