import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt


class PretrainedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        ResNet = models.resnet18(pretrained=True)
        ResNet.fc = nn.Linear(ResNet.fc.in_features, len(VOCDataset.CLASS_NAMES))
        nn.init.xavier_normal_(ResNet.fc.weight)
        self.model = ResNet
    
    def forward(self, x):
        return self.model(x)

class RetrieveResnet(nn.Module):
  def __init__(self):
    super(RetrieveResnet, self).__init__()

    self.res1 = models.resnet50(pretrained=True)
    self.res2 = models.resnet50(pretrained=True)

    self.res1.fc = nn.Linear(self.res1.fc.in_features, len(VOCDataset.CLASS_NAMES))
        nn.init.xavier_normal_(ResNet.fc.weight)

    self.conv = nn.Conv2d( ... )  # set up your layer here
    self.fc1 = nn.Linear( ... )  # set up first FC layer
    self.fc2 = nn.Linear( ... )  # set up the other FC layer

  def forward(self, input1, input2):
    c = self.conv(input1)
    f = self.fc1(input2)
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
    out = self.fc2(combined)
    return out