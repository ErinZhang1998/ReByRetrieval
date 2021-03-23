from train import *
from utils.plot_image import *

args = this_utils.ARGS(epochs=100, batch_size=32, lr=0.0001, val_every=100, \
            save_freq=5, save_at_end=True, use_cuda = True)

model = resnet_pretrain.PretrainedResNet()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=args.gamma)
train(args, model, optimizer, scheduler=scheduler, model_name='resnet50', dataset_size = 227)

# train_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set', train=True, batch_size=args.batch_size, \
#                                          split='train', \
#                                         dataset_size = 227)
# test_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/testing_set', train=False, batch_size=args.test_batch_size, \
#                                     split='test', \
#                                     dataset_size = 227)

# dataset = InCategoryClutterDataset('train', 227, '/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set', \
#     '/media/xiaoyuz1/hdd5/xiaoyuz1/data/taxonomy_tabletop_small_keys.txt')

# sample = dataset.idx_to_data_dict[1000]
# # trans = utrans.Compose([utrans.Resized(width = int(dataset.size * 1.5), height = int(dataset.size * 1.5)),
# #         utrans.RandomCrop(size = int(dataset.size)),
# #         utrans.RandomHorizontalFlip(),
# #     ])
# trans = utrans.Compose([utrans.Resized(width = dataset.size, height = dataset.size),
#         utrans.RandomHorizontalFlip(),
#     ])

# rgb_all = mpimg.imread(sample['rgb_all_path'])
# # mask = (mpimg.imread(sample['mask_path']) > 0).astype('int')
# # mask = np.stack([mask,mask,mask],axis=2) #np.expand_dims(mpimg.imread(sample['mask_path']), axis=0)
# mask = mpimg.imread(sample['mask_path'])
# center = copy.deepcopy(sample['object_center'].reshape(-1,))
# center[0] = rgb_all.shape[1] - center[0]


# img_rgb, center_trans = trans(rgb_all, center)
# img_mask, center_trans = trans(mask , center)

# img_rgb = utrans.normalize(utrans.to_tensor(img_rgb), [0.5,0.5,0.5], [0.5,0.5,0.5])

# img = torch.cat((img_rgb, img_mask[:1,:,:]), 0)
# image = torch.FloatTensor(img)

# #pose_info = np.concatenate((np.array([sample['scale']]).reshape(-1,), sample['orientation'].reshape(-1,), sample['object_center'].reshape(-1,)))

# scale_info = torch.FloatTensor(np.array([sample['scale']]).reshape(-1,))
# orient_info = torch.FloatTensor(sample['orientation'].reshape(-1,))
# pixel_info = torch.FloatTensor(center_trans.reshape(-1,) / dataset.size)
# cat_info = torch.FloatTensor(np.array([sample['obj_cat']]).reshape(-1,))

# #return image, torch.FloatTensor(pose_info), torch.FloatTensor([sample['obj_cat']])
# return image, scale_info, orient_info, pixel_info, cat_info