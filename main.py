import yaml 
from optparse import OptionParser
import torch
import wandb
import os 

import resnet_pretrain
import utils.utils as uu
import train
import incat_dataset
import incat_dataloader

parser = OptionParser()
parser.add_option("--config_file", dest="config_file")
(options, args) = parser.parse_args()

f =  open(options.config_file)
args_dict = yaml.safe_load(f)
default_args_dict = yaml.safe_load(open('configs/default.yaml'))
args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)

wandb.login()
wandb.init(project='erin_retrieval', entity='erin_retrieval', config=args_dict_filled)

args = uu.Struct(args_dict_filled)
device = torch.device("cuda" if args.use_cuda else "cpu")
model = resnet_pretrain.PretrainedResNet(emb_dim=args.model_config.emb_dim, pose_dim=args.model_config.pose_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer_config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_config.step, gamma=args.scheduler_config.gamma)

train_dataset = incat_dataset.InCategoryClutterDataset('train', args.dataset_config.size, \
    scene_dir=args.files.training_scene_dir, \
    model_filepath=args.files.training_model_filepath, \
    shape_categories_filepath=args.files.shape_categories_filepath, \
    shapenet_filepath=args.files.shapenet_filepath)
train_loader = incat_dataloader.InCategoryClutterDataloader(train_dataset, args.training_config.batch_size, shuffle = True)

test_dataset = incat_dataset.InCategoryClutterDataset('test', args.dataset_config.size, \
    scene_dir=args.files.testing_scene_dir, \
    model_filepath=args.files.testing_model_filepath, \
    shape_categories_filepath=args.files.shape_categories_filepath, \
    shapenet_filepath=args.files.shapenet_filepath)
test_loader = incat_dataloader.InCategoryClutterDataloader(test_dataset, args.testing_config.batch_size, shuffle = False)

trainer = train.Trainer(args, model, train_loader, test_loader, optimizer, scheduler, device)
trainer.train()

# model = resnet_pretrain.PretrainedResNet()
# # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=args.gamma)
# # train(args, model, optimizer, scheduler=scheduler, model_name='resnet50', dataset_size = 227)

# # train_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set', train=True, batch_size=args.batch_size, \
# #                                          split='train', \
# #                                         dataset_size = 227)

# dataset = InCategoryClutterDataset('train', 227, '/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set', \
#     '/media/xiaoyuz1/hdd5/xiaoyuz1/data/taxonomy_tabletop_small_keys.txt')
# train_loader = InCategoryClutterDataloader(dataset, args.batch_size)
# image, scale_info, pixel_info, cat_info, id_info = None,None,None,None,None
# for batch_idx, (image, scale_info, pixel_info, cat_info, id_info) in enumerate(train_loader):
#     break

# model = model.to(device)
# image = image.to(device)
# scale_info = scale_info.to(device)
# pixel_info = pixel_info.to(device)
# cat_info = cat_info.to(device)
# id_info = id_info.to(device)

# # optimizer.zero_grad()
# img_embed, pose_pred = model(image)
# # anchor, positives, negatives = selector(img_embed, cat_info.view(-1,))
# embeddings = img_embed
# pairwise_dist = this_utils.pariwise_distances(embeddings, squared=squared)


# anchor_positive_dist = torch.unsqueeze(pairwise_dist, dim=2)
# anchor_negative_dist = torch.unsqueeze(pairwise_dist, dim=1)
# triplet_loss = anchor_positive_dist - anchor_negative_dist + 1.0 

# la_not_lp = labels != labels.T
# la_is_ln = labels == labels.T
# a = torch.arange(len(embeddings)).view(-1,1).to(embeddings.device)
# a_is_p = a == a.T
# la_not_lp = torch.unsqueeze(la_not_lp, dim=2)
# la_is_ln = torch.unsqueeze(la_is_ln, dim=1)
# mask_1 = torch.logical_or(la_not_lp, la_is_ln)
# mask_2 = torch.stack(len(embeddings)*[a_is_p], dim=0) 
# mask = torch.logical_not(torch.logical_or(mask_1,mask_2))
# mask = mask.float()

# triplet_loss = mask * triplet_loss
# triplet_loss = torch.maximum(triplet_loss, torch.zeros_like(triplet_loss))

# valid_triplets = torch.gt(triplet_loss, torch.ones_like(triplet_loss)*1e-16).float()
# num_positive_triplets = torch.sum(valid_triplets)
# num_valid_triplets = torch.sum(mask)
# fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

# triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)



# # test_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/testing_set', train=False, batch_size=args.test_batch_size, \
# #                                     split='test', \
# #                                     dataset_size = 227)



# # sample = dataset.idx_to_data_dict[1000]
# # # trans = utrans.Compose([utrans.Resized(width = int(dataset.size * 1.5), height = int(dataset.size * 1.5)),
# # #         utrans.RandomCrop(size = int(dataset.size)),
# # #         utrans.RandomHorizontalFlip(),
# # #     ])
# # trans = utrans.Compose([utrans.Resized(width = dataset.size, height = dataset.size),
# #         utrans.RandomHorizontalFlip(),
# #     ])

# # rgb_all = mpimg.imread(sample['rgb_all_path'])
# # # mask = (mpimg.imread(sample['mask_path']) > 0).astype('int')
# # # mask = np.stack([mask,mask,mask],axis=2) #np.expand_dims(mpimg.imread(sample['mask_path']), axis=0)
# # mask = mpimg.imread(sample['mask_path'])
# # center = copy.deepcopy(sample['object_center'].reshape(-1,))
# # center[0] = rgb_all.shape[1] - center[0]


# # img_rgb, center_trans = trans(rgb_all, center)
# # img_mask, center_trans = trans(mask , center)

# # img_rgb = utrans.normalize(utrans.to_tensor(img_rgb), [0.5,0.5,0.5], [0.5,0.5,0.5])

# # img = torch.cat((img_rgb, img_mask[:1,:,:]), 0)
# # image = torch.FloatTensor(img)

# # #pose_info = np.concatenate((np.array([sample['scale']]).reshape(-1,), sample['orientation'].reshape(-1,), sample['object_center'].reshape(-1,)))

# # scale_info = torch.FloatTensor(np.array([sample['scale']]).reshape(-1,))
# # orient_info = torch.FloatTensor(sample['orientation'].reshape(-1,))
# # pixel_info = torch.FloatTensor(center_trans.reshape(-1,) / dataset.size)
# # cat_info = torch.FloatTensor(np.array([sample['obj_cat']]).reshape(-1,))

# # #return image, torch.FloatTensor(pose_info), torch.FloatTensor([sample['obj_cat']])
# # return image, scale_info, orient_info, pixel_info, cat_info