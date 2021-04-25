import yaml 
from optparse import OptionParser
import torch
import wandb
import os 

import resnet_pretrain
import utils.utils as uu
import train
import test 
import incat_dataset
import incat_dataloader

parser = OptionParser()
parser.add_option("--config_file", dest="config_file")
parser.add_option("--only_test", dest="only_test", action='store_true')
parser.add_option("--model_path", dest="model_path", default='')
(options, args) = parser.parse_args()

f =  open(options.config_file)
# f = open('configs/config_all_occlusion.yaml')
args_dict = yaml.safe_load(f)
default_args_dict = yaml.safe_load(open('configs/default.yaml'))
args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)

if not options.only_test:
    wandb.login()
    wandb.init(project='erin_retrieval', entity='erin_retrieval', config=args_dict_filled)

args = uu.Struct(args_dict_filled)
device = torch.device("cuda" if args.use_cuda else "cpu")
model = resnet_pretrain.PretrainedResNet(emb_dim=args.model_config.emb_dim, pose_dim=args.model_config.pose_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer_config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_config.step, gamma=args.scheduler_config.gamma)

train_dataset = incat_dataset.InCategoryClutterDataset('train', args)
train_loader = incat_dataloader.InCategoryClutterDataloader(train_dataset, args.training_config.batch_size, shuffle = True)

test_dataset = incat_dataset.InCategoryClutterDataset('test', args)
test_loader = incat_dataloader.InCategoryClutterDataloader(test_dataset, args.testing_config.batch_size, shuffle = False)

if options.only_test:
    if args.model_config.model_path is None:
        if options.model_path == '':
            print("ERROR: when only testing, must provide model_path in model_config to retrieve the model")
            exit(0)
        else:
            args.model_config.model_path = options.model_path

trainer = train.Trainer(args, model, train_loader, test_loader, optimizer, scheduler, device)
trainer.train(options.only_test)