import torch
from torch import nn

def train(model, device, train_loader, optimizer, epoch):
    criterion = nn.BCELoss()
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    # print(total_num, len(train_loader))
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        output = model(imgs)
        loss = criterion(output, targets.type(torch.float)) # criterion = nn.BCELoss()

        print_loss = loss.item()

        loss.backward()
        optimizer.step()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(imgs), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

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
    total_num = len(test_loader.dataset)
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
