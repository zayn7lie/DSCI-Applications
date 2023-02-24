import torch
from torch import nn

def adjust_lr(optimizer, epoch, modellr):
    modellrnew = modellr * (0.1 ** (epoch // 25))
    print("lr:", modellrnew, "\n\n")
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
        
        if (batch_idx + 1) % 20 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: MAX={:.6f} MIN={:.6f} AVG={:.6f}'.format(
                epoch, (batch_idx + 1) * len(imgs), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), maxloss, minloss, sumloss / 20))
            sumloss = 0
            minloss = 1
            maxloss = 0

    avg_loss = sum_loss / len(train_loader)
    print('Epoch: {}\tLoss:{}\t'.format(epoch, avg_loss))

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def eval(model, device, test_loader):
    criterion = nn.BCELoss()
    model.eval()
    test_loss = 0
    kappa = 0
    f1 = 0
    auc = 0
    # total_num = len(test_loader.dataset)
    # print(total_num, len(test_loader))
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            output = model(imgs)
            loss = criterion(output, targets.type(torch.float))
            
            print_loss = loss.data.item()
            test_loss += print_loss

            output, targets = output.cpu().numpy(), targets.cpu().numpy()
            for i in range(output):
                if output[i] >= 0.5: output[i] = 1
                else: output[i] = 0
            kappa += cohen_kappa_score(targets, output)
            f1 += f1_score(targets, output)
            auc += roc_auc_score(targets, output)
        avgloss = 1.000 * test_loss / len(test_loader)
        avgkappa = 1.000 * kappa / len(test_loader)
        avgf1 = 1.000 * f1 / len(test_loader)
        avgauc = 1.000 * auc / len(test_loader)
        print('\nVal set: Average loss: {:.4f} Average kappa: {:.4f} Average f1: {:.4f} Average auc: {:.4f}\n', avgloss, avgkappa, avgf1, avgauc)
