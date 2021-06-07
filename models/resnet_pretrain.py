import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from .build import MODEL_REGISTRY

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints
        
@MODEL_REGISTRY.register()
class PretrainedResNetSpatialSoftmax(nn.Module):
    def __init__(self, args):
        super(PretrainedResNetSpatialSoftmax, self).__init__()
        self.emb_dim=args.model_config.emb_dim
        self.pose_dim=args.model_config.pose_dim
        res50 = models.resnet50(pretrained=True)
        res50.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=False)
        self.res50_no_fc = nn.Sequential(*list(res50.children())[:-2])

        self.spatial_softmax = SpatialSoftmax(9,12,2048)
        
        self.emb_fc = nn.Linear(res50.fc.in_features*2, self.emb_dim)
        self.pose_fc = nn.Linear(res50.fc.in_features*2, self.pose_dim)
    
    def forward(self, x):
        x = self.res50_no_fc(x)
        x = self.spatial_softmax(x)
        emb = self.emb_fc(x)
        pose = self.pose_fc(x)
        return emb, pose

@MODEL_REGISTRY.register()
class PretrainedResNet(nn.Module):
    def __init__(self, args):
        super(PretrainedResNet, self).__init__()
        self.emb_dim=args.model_config.emb_dim
        res50 = models.resnet50(pretrained=True)
        res50.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=False)
        self.res50_no_fc = nn.Sequential(*list(res50.children())[:-1])
        self.flat_dim = res50.fc.in_features
        self.emb_fc = nn.Linear(res50.fc.in_features, self.emb_dim)
        self.pose_fc = nn.Linear(res50.fc.in_features, self.pose_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.res50_no_fc(x)
        flat_x = x.view(batch_size, self.flat_dim)
        emb = self.emb_fc(flat_x)
        pose = self.pose_fc(flat_x)
        return emb, pose