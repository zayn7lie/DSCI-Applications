import torch
from torch import nn

def adjust_lr(optimizer, epoch, modellr):
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("Epoch:", epoch, "Learning Rate:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

def train(model, device, train_loader, optimizer, epoch):
    criterion = nn.BCELoss()
    model.train()
    sum_loss = 0
    sumloss = 0
    minloss = 1
    maxloss = 0
    # total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        output = model(imgs)
        loss = criterion(output, targets.type(torch.float)) # criterion = nn.BCELoss()

        print_loss = loss.item()
        minloss = min(minloss, print_loss)
        maxloss = max(maxloss, print_loss)
        sum_loss += print_loss
        sumloss += print_loss

        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 5 == 0:
            print("- [{}/{} ({:.0f}%)] Loss: AVG={:.6f} MAX={:.6f} MIN={:.6f}".format((batch_idx + 1) * len(imgs), len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), sumloss / 5, maxloss, minloss))
            sumloss = 0
            minloss = 1
            maxloss = 0

    avg_loss = sum_loss / len(train_loader)
    print("Loss:", avg_loss, '\n')

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy

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
            kappa += cohen_kappa_score(targets, output)
            f1 += f1_score(targets, output)
            auc += roc_auc_score(targets, output)
            print(kappa, f1, auc)
        avgloss = test_loss / cnt
        avgkappa = kappa / cnt
        avgf1 = f1 / cnt
        avgauc = auc / cnt
        print('\nVal set: Average loss: {:.4f} Average kappa: {:.4f} Average f1: {:.4f} Average auc: {:.4f}\n'.format(avgloss, avgkappa, avgf1, avgauc))
