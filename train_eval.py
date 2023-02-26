import torch
from torch import nn

def adjust_lr(optimizer, epoch, modellr):
    modellrnew = modellr * (0.1 ** (epoch // 41)) # 25
    print("Epoch:", epoch, "Learning Rate:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def train(epoch, model, device, tr_loader_x, tr_loader_y, optimizer):
    criterion = nn.BCELoss()
    model.train()
    sum_loss, sum_mmd, sum_bce = 0, 0, 0
    cnt = 1
    sumloss, minloss, maxloss = 0, 1, 0
    iter_y = iter(tr_loader_y)
    
    for batch_idx, (imgs_x, targets) in enumerate(tr_loader_x):
        imgs_x, targets = imgs_x.to(device), targets.to(device)
        
        imgs_y, _ = iter_y.__next__()
        imgs_y = imgs_y.to(device)
        cnt += 1
        if cnt % len(tr_loader_x) == 0:
            iter_y = iter(tr_loader_y)

        optimizer.zero_grad()

        output, mmd_loss = model(imgs_x, imgs_y)
        bce_loss = criterion(output, targets.type(torch.float))
        loss = bce_loss + 0.025 * mmd_loss
        
        sumloss += loss
        minloss, maxloss = min(minloss, loss), max(maxloss, loss)
        
        sum_loss += loss
        sum_mmd += mmd_loss
        sum_bce += bce_loss

        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 9 == 0:
            print("- [{:.0f}/{:.0f}] Loss: AVG={:.6f} MAX={:.6f} MIN={:.6f}".format((batch_idx + 1), len(tr_loader_x), sumloss / 9, maxloss, minloss))
            sumloss, minloss, maxloss = 0, 1, 0

    avg_loss, avg_mmd, avg_bce = sum_loss / len(tr_loader_x), sum_mmd / len(tr_loader_x), sum_bce / len(tr_loader_x)
    print("Epoch:", epoch, "Loss:", avg_loss, "MMD:", avg_mmd, "BCE:", avg_bce, '\n')

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def eval(model, device, test_loader):
    criterion = nn.BCELoss()
    model.eval()
    test_loss, cnt = 0, 0
    kappa, f1, auc = 0, 0, 0
    # total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    with torch.no_grad():
        for imgs, targets in test_loader:
            cnt += 1
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)
            loss = criterion(output, targets.type(torch.float))
            
            print_loss = loss.data.item()
            test_loss += print_loss

            output = output.cpu().numpy().flat
            targets = targets.cpu().numpy().flat
            for i in range(len(output)):
                if output[i] >= 0.5: output[i] = 1
                else: output[i] = 0
            f1 += f1_score(targets, output)
            auc += roc_auc_score(targets, output)
            print(kappa, f1, auc)
        avgloss = test_loss / cnt
        avgf1 = f1 * 100 / cnt
        avgauc = auc * 100 / cnt
        print('\nVal set: Average loss: {:.4f} Average f1: {:.4f}% Average auc: {:.4f}%\n'.format(avgloss, avgf1, avgauc))
