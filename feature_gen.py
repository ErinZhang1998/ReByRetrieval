import incat_dataset
import incat_dataloader
import yaml 
from optparse import OptionParser
import utils.utils as uu
import torch
import resnet_pretrain
import wandb
import os 
import numpy as np

parser = OptionParser()
parser.add_option("--root_dir", default='/raid/xiaoyuz1/models/')
parser.add_option("--experiment_name", dest="experiment_name")
parser.add_option("--epoch", type=int, dest="epoch")
parser.add_option("--config_file", dest="config_file")
parser.add_option("--output_dir", dest="output_dir")

(options, args) = parser.parse_args()

f =  open(options.config_file)
args_dict = yaml.safe_load(f)

args = uu.Struct(args_dict)
device = torch.device('cuda')

model = resnet_pretrain.PretrainedResNet(emb_dim=args.model_config.emb_dim, pose_dim=args.model_config.pose_dim)

model_path = os.path.join(options.root_dir, options.experiment_name, '{}.pth'.format(options.epoch))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = incat_dataset.InCategoryClutterDataset('test', args.dataset_config.size, \
    scene_dir=args.files.testing_scene_dir, \
    model_filepath=args.files.testing_model_filepath, \
    shape_categories_filepath=args.files.shape_categories_filepath, \
    shapenet_filepath=args.files.shapenet_filepath)

test_loader = incat_dataloader.InCategoryClutterDataloader(test_dataset, args.testing_config.batch_size, shuffle = False)

embeds = []
scene_names_list = []

pose_pred_list = []
pose_info_list = []


with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        # if batch_idx % 50 == 0:
        print("=> Extracting ", batch_idx)
        image = data[0]
        scale_info = data[1]
        pixel_info = data[2]
        index = data[-1]  

        dataset_indices = data[5]
        # ind = list(dataset_indices.numpy().astype(int).reshape(-1,))
        scene_names = test_dataset.idx_to_sample_id[dataset_indices.numpy().astype(int)].reshape(-1,)
        scene_names_list.append(scene_names)

        model = model.to(device)
        image = image.to(device)

        img_embed, pose_pred = model(image)
        img_embed = img_embed.cpu()
        pose_pred = pose_pred.cpu()
        pose_pred = pose_pred.float().detach()

        pose_pred_list.append(pose_pred)
        assert torch.cat([scale_info, pixel_info], dim=1).shape[1] == 3
        pose_info_list.append(torch.cat([scale_info, pixel_info], dim=1))
        
        embeds.append(img_embed)
        
        torch.cuda.empty_cache()

all_embedding = torch.cat(embeds, dim=0).numpy()

try:
    pose_pred = torch.cat(pose_pred_list, dim=0)
    pose_info = torch.cat(pose_info_list, dim=0)
    all_pose = torch.cat([pose_pred, pose_info], dim=1)

    pose_path = os.path.join(options.output_dir, '{}_{}_pose.npy'.format(options.experiment_name, options.epoch))
    np.save(pose_path, all_pose)
except:
    print("Cannot output pose information!")

feat_path = os.path.join(options.output_dir, '{}_{}.npy'.format(options.experiment_name, options.epoch))
np.save(feat_path, all_embedding)

all_scene_names = np.hstack(scene_names_list)
scane_name_path = os.path.join(options.output_dir, '{}_{}_scenes.npy'.format(options.experiment_name, options.epoch))
np.save(scane_name_path, all_scene_names)
# python feature_gen.py --experiment_name absurd-waterfall-22 --epoch 25 --config_file configs/config_all_1.yaml --output_dir /raid/xiaoyuz1/retrieve_features