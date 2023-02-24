from torch import nn
from torchvision import models
class Resnet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet50()
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=8)
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

class RMMD(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnet50()
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=8)
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))
