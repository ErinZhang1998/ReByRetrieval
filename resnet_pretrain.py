import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt


class PretrainedResNet(nn.Module):
    def __init__(self, emb_dim = 128, pose_dim = 3):
        super().__init__()
        res50 = models.resnet50(pretrained=True)
        res50.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=False)
        self.res50_no_fc = nn.Sequential(*list(res50.children())[:-1])
        # self.emb_fc = nn.Sequential( 
        #               nn.Linear(res50.fc.in_features, 1024), 
        #               nn.BatchNorm2d(1024),
        #               nn.ReLU(),
        #               nn.Linear(1024, emb_dim)) 
        self.flat_dim = res50.fc.in_features
        self.emb_fc = nn.Linear(res50.fc.in_features, emb_dim)
        self.pose_fc = nn.Linear(res50.fc.in_features, pose_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.res50_no_fc(x)
        flat_x = x.view(batch_size, self.flat_dim)
        emb = self.emb_fc(flat_x)
        pose = self.pose_fc(flat_x)
        return emb, pose

# class RetrieveResnet(nn.Module):
#   def __init__(self):
#     super(RetrieveResnet, self).__init__()

#     self.res1 = models.resnet50(pretrained=True)
#     self.res2 = models.resnet50(pretrained=True)

#     self.res1.fc = nn.Linear(self.res1.fc.in_features, len(VOCDataset.CLASS_NAMES))
#         nn.init.xavier_normal_(ResNet.fc.weight)

#     self.conv = nn.Conv2d( ... )  # set up your layer here
#     self.fc1 = nn.Linear( ... )  # set up first FC layer
#     self.fc2 = nn.Linear( ... )  # set up the other FC layer

#   def forward(self, input1, input2):
#     c = self.conv(input1)
#     f = self.fc1(input2)
#     # now we can reshape `c` and `f` to 2D and concat them
#     combined = torch.cat((c.view(c.size(0), -1),
#                           f.view(f.size(0), -1)), dim=1)
#     out = self.fc2(combined)
#     return out