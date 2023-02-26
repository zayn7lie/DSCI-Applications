import torch
from torch import nn
from torchvision import models

class Resnet50(models.ResNet):
    def __init__(self):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=8)
        self.sigm = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)        
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.sigm(x), 0

class RMMD(Resnet50):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(716800 * 14, 256),
            nn.ReLU(inplace=False)
        )
    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # ResNet-50 with MMD
        x_ = self.bottleneck(x)
        mmd_loss = 0
        if self.training:
            y = self.conv1(y)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.maxpool(y)

            y = self.layer1(y)
            y = self.layer2(y)
            y = self.layer3(y)
            y_ = self.bottleneck(y)
            print(y_)
            print(x_)
            print(x_ - y_)
            mmd_loss += torch.mean(torch.mm(x_ - y_, torch.transpose(x_ - y_, 0, 1)))
        
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return self.sigm(x), mmd_loss
