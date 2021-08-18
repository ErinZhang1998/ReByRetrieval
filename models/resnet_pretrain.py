import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
from .resnet import resnet18, resnet50
import matplotlib.pyplot as plt
import numpy as np
from .build import MODEL_REGISTRY


from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from pointnet2_ops import pointnet2_utils

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
        self.args = args
        self.emb_dim = args.model_config.emb_dim
        self.pose_dim = args.model_config.pose_dim
        
        if args.model_config.resnet_type == 'resnet18':
            resnet = resnet18(pretrained=True)
        elif args.model_config.resnet_type == 'resnet50':
            resnet = resnet50(pretrained=True, remove_avg_pool_layer=True)
        self.resnet_out_channel = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=False)
        self.resnet_no_fc = resnet #nn.Sequential(*list(resnet.children())[:-2])

        self.ss = args.model_config.spatial_softmax
        self.spatial_softmax = SpatialSoftmax(self.ss.height, self.ss.width, self.resnet_out_channel)


        return_keys = list(set(args.model_config.model_return))
        # ['class_pred', 'scale_pred', 'center_pred']
        if 'class_pred' in return_keys:
            df  = pd.read_csv(args.files.csv_file_path)
            if args.model_config.class_type == 'shapenet_model_id':
                num_classes = len(df)
            elif args.model_config.class_type == 'obj_category':
                num_classes = len(df['catId'].unique())
            else:
                num_classes = len(df['objId'].unique())
            self.add_module('class_pred', nn.Linear(self.resnet_out_channel*2, num_classes))
        
        if 'scale_pred' in return_keys:
            self.add_module('scale_pred', nn.Linear(self.resnet_out_channel*2, 1))
        
        if 'center_pred' in return_keys:
            self.add_module('center_pred', nn.Linear(self.resnet_out_channel*2, 2))
        
        if 'img_embed' in return_keys:
            self.add_module('img_embed', nn.Linear(self.resnet_out_channel*2, self.emb_dim))
        self.return_keys = return_keys
            
    def forward(self, xs):
        x = xs[0]
        x = self.resnet_no_fc(x)
        x = self.spatial_softmax(x)
        
        return_dict = {}
        for return_key in self.return_keys:
            head = getattr(self, return_key)
            pred = head(x)
            if return_key == 'img_embed':
                pred -= pred.min(1, keepdim=True)[0]
                pred /= pred.max(1, keepdim=True)[0]
            return_dict[return_key] = pred
        return return_dict
        # if self.args.model_config.classification:
        #     return {
        #         'class_pred' : self.classification_head(x), 
        #         'pose_pred': self.pose_fc(x),
        #     }
        # else:
        #     emb = self.emb_fc(x)
        #     pose = self.pose_fc(x)
        #     return {
        #         'img_embed': emb, 
        #         'pose_pred' : pose,
        # }

@MODEL_REGISTRY.register()
class PretrainedResNet(nn.Module):
    def __init__(self, args):
        super(PretrainedResNet, self).__init__()
        self.emb_dim=args.model_config.emb_dim
        if args.model_config.resnet_type == 'resnet18':
            resnet = resnet18(pretrained=True)
        elif args.model_config.resnet_type == 'resnet50':
            resnet = resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=False)
        self.resnet_no_fc = nn.Sequential(*list(resnet.children())[:-1])
        self.flat_dim = resnet.fc.in_features
        self.emb_fc = nn.Linear(resnet.fc.in_features, self.emb_dim)
        self.pose_fc = nn.Linear(resnet.fc.in_features, self.pose_dim)
    
    def forward(self, xs):
        x = xs[0]
        batch_size = x.size(0)
        x = self.resnet_no_fc(x)
        flat_x = x.view(batch_size, self.flat_dim)
        emb = self.emb_fc(flat_x)
        pose = self.pose_fc(flat_x)
        return {
            'img_embed': emb, 
            'pose_pred' : pose,
        }

@MODEL_REGISTRY.register()
class ResNetPointNet(nn.Module):
    def __init__(self, args):
        super(ResNetPointNet, self).__init__()
        self.emb_dim=args.model_config.emb_dim
        self.pose_dim=args.model_config.pose_dim
        if args.model_config.resnet_type == 'resnet18':
            resnet = resnet18(pretrained=True)
        elif args.model_config.resnet_type == 'resnet50':
            resnet = resnet50(pretrained=True)
        self.resnet_out_channel = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0), bias=False)
        self.resnet_no_fc = resnet#nn.Sequential(*list(resnet.children())[:-2])
        
        self.ss = args.model_config.spatial_softmax
        self.spatial_softmax = SpatialSoftmax(self.ss.height, self.ss.width, self.resnet_out_channel)

        self.pc_points_per_obj = args.model_config.pointnet.pc_points_per_obj
        self.msgmodules = args.model_config.pointnet.msgmodules
        self.samodules = args.model_config.pointnet.samodules

        self.SA_modules = nn.ModuleList()
        input_channels = []
        for i in range(len(self.msgmodules.npoints)):
            if i == 0:
                mlp_shape = self.msgmodules.mlpss[0]
            else:
                mlp_shape = [[input_channels[-1]] + l for l in self.msgmodules.mlpss[i]]
            input_channels.append(sum([j[-1] for j in self.msgmodules.mlpss[i]]))
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint = self.msgmodules.npoints[i],
                    radii = self.msgmodules.radiis[i],
                    nsamples = self.msgmodules.nsampless[i],
                    mlps = mlp_shape,
                    use_xyz=True,
                )
            )

        mlp_shape = [input_channels[-1]] + self.samodules.mlpss
        self.SA_modules.append(
            PointnetSAModule(
                mlp=mlp_shape,
                use_xyz=True,
            )
        )
        
        self.emb_fc = nn.Linear(self.resnet_out_channel*2 + mlp_shape[-1], self.emb_dim)
        self.pose_fc = nn.Linear(self.resnet_out_channel*2 + mlp_shape[-1], self.pose_dim)
    
    def forward(self, xs):
        '''
        xs = [image, pts, feats]
        image : (B, 3, H, W)
        pts : (B, N, 3)
        feats : (B, C, N)
        '''
        image = xs[0]
        pts, feats = xs[1:]  
        sampled_pts = pointnet2_utils.furthest_point_sample(pts, self.pc_points_per_obj)
        pts_flipped = pts.transpose(1, 2).contiguous()
        xyz = pointnet2_utils.gather_operation(pts_flipped, sampled_pts).transpose(1, 2).contiguous()
        features = pointnet2_utils.gather_operation(feats, sampled_pts)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        features = features.squeeze(-1)
        x = self.resnet_no_fc(image)
        x = self.spatial_softmax(x)
        
        x_cat = torch.cat([x,features], dim=1)
        
        emb = self.emb_fc(x_cat)
        pose = self.pose_fc(x_cat)
        return emb, pose