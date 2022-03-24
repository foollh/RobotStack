import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

    def forward(self, x):
        
        pass

if __name__ == "__main__":
    resnet50 = models.resnet50(pretrained=False)
    outLayer = nn.Sequential(nn.Linear(resnet50.fc.in_features, 200, True),
                nn.Linear(200, 7, True))
    resnet50.fc = outLayer

