''' change working directory 
import os

os.chdir("C:/Users/zayn7lie/OneDrive - ber7/02-Prog/GitHub/ML-RMMD")
print('current working directory is {}'.format(os.getcwd()))

print(os.getcwd())
print(os.listdir(os.getcwd()))
'''

import numpy as np
# import ResNet

# normalize(preopera) figures
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
Transform = transforms.Compose([
    transforms.Resize(512), # short l -> 2048
    transforms.CenterCrop(512), # crop
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# import xlrd table to numpy.array
import xlrd
def rData(filepath):
    data = xlrd.open_workbook(filepath)
    table = data.sheets()[0]
    data = [ [int(table.row_values(i,0,1)[0])] + table.row_values(i,-8) for i in range(1,table.nrows)]
    return np.array(data)
    # data: num, bool: NDGCAHMO

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.utils.data

BATCH_SIZE = 16
EPOCHS = 10
modellr = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        # correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))

def main():

    # import data
    trainLabel = rData("./OIA-ODIR/Training Set/Annotation/training annotation (English).xlsx").flatten() # data: num, bool: NDGCAHMO
    trainSet = datasets.ImageFolder(root="./OIA-ODIR/Training Set/Images", transform=Transform)
    trainLoader = DataLoader(dataset=trainSet, batch_size=8, shuffle=True, num_workers=8) 

    testLabel = rData("./OIA-ODIR/Off-site Test Set/Annotation/off-site test annotation (English).xlsx").flatten()
    testSet = datasets.ImageFolder(root="./OIA-ODIR/Off-site Test Set/Images", transform=Transform)
    testLoader = DataLoader(dataset=testSet, batch_size=8, shuffle=True, num_workers=8)

    # model building
    model = torchvision.models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=modellr)

    # start training
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train(model, DEVICE, trainLoader, optimizer, epoch)
        val(model, DEVICE, testLoader)
        torch.save(model, 'model.pth')

    print("finished")

if __name__ == "__main__":
    main()
