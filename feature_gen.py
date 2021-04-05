import train
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
indices = []
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        # if batch_idx % 50 == 0:
        print("=> Extracting ", batch_idx)
        image = data[0]
        index = data[4]  

        model = model.to(device)
        image = image.to(device)

        img_embed, _ = model(image)
        img_embed = img_embed.cpu()
        
        embeds.append(img_embed)
        indices.append(index)
        
        torch.cuda.empty_cache()

all_embedding = torch.cat(embeds, dim=0).numpy()
all_indices = torch.cat(indices, dim=0).numpy()

feat_path = os.path.join(options.output_dir, '{}_{}.npy'.format(options.experiment_name, options.epoch))
np.save(feat_path, all_embedding)

ind_path = os.path.join(options.output_dir, '{}_{}_index.npy'.format(options.experiment_name, options.epoch))
np.save(ind_path, all_indices)
# python feature_gen.py --experiment_name volcanic-eon-17 --epoch 5 --config_file configs/config_only_cat_obj.yaml --output_dir /raid/xiaoyuz1/retrieve_features