from __future__ import print_function

import torch
import numpy as np
import os
import torchvision
import PIL
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils.plot_image as uplot 
import utils.transforms as utrans
import utils.utils as uu
import utils.distributed as du
import utils.logging as logging
from utils.meters import TestMeter, FeatureExtractMeter

import losses.loss as loss

from models.build import build_model
import incat_dataset
import incat_dataloader
import test

logger = logging.get_logger(__name__)

def save_this_epoch(args, epoch):

    if args.save_freq < 0:
        return False 
    return epoch % args.save_freq == 0


def train_epoch(args, train_loader, model, optimizer, epoch, cnt, image_dir=None, wandb_enabled=False):
    model.train()

    training_config = args.training_config
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        image = data['image'].cuda(non_blocking=args.cuda_non_blocking)
        return_keys, return_val = model([image])

        if 'scale_pred' in return_keys:
            scale_pred = return_val[return_keys.index('scale_pred')]
            scale = data['scale'].cuda()
            scale_loss = loss.get_loss_func(args.loss.scale_pred_fn)(scale_pred, scale) * args.loss.lambda_scale_pred

        if 'center_pred' in return_keys:
            center_pred = return_val[return_keys.index('center_pred')]
            center = data['center'].cuda()
            center_loss = loss.get_loss_func(args.loss.center_pred_fn)(center_pred, center) * args.loss.lambda_center_pred
        
        if 'class_pred' in return_keys:
            class_pred = return_val[return_keys.index('class_pred')]
            class_gt = data[args.model_config.class_type].cuda()
            class_loss = loss.get_loss_func(args.loss.class_pred_fn)(class_pred, class_gt.view(-1,).long()) * args.loss.lambda_class_pred
        
        if 'img_embed' in return_keys:
            img_embed = return_val[return_keys.index('img_embed')]
            obj_category = data['obj_category'].cuda()
            obj_id = data['obj_id'].cuda()

            triplet_mask_obj_category, obj_category_triplet_loss = loss.batch_all_triplet_loss(
                labels = obj_category, 
                embeddings = img_embed, 
                margin = args.loss.margin, 
                squared=False,
            )
            triplet_mask_obj_id, obj_id_triplet_loss = loss.batch_all_triplet_loss(
                labels = obj_id, 
                embeddings = img_embed, 
                margin = args.loss.margin, 
                squared=False,
            )    
            obj_category_triplet_loss = obj_category_triplet_loss * args.loss.lambda_obj_category
            obj_id_triplet_loss = obj_id_triplet_loss * args.loss.lambda_obj_id

        total_set = False
        if 'img_embed' in return_keys:
            total_loss = obj_category_triplet_loss + obj_id_triplet_loss
            total_set = True
        if 'class_pred' in return_keys:
            if total_set:
                total_loss += class_loss
            else:
                total_loss = class_loss
                total_set = True
        
        if 'scale_pred' in return_keys:
            total_loss += scale_loss

        if 'center_pred' in return_keys:
            total_loss += center_loss

        assert total_set

        total_loss.backward()
        optimizer.step()

        if args.num_gpus > 1:
            total_loss = du.all_reduce([total_loss])[0]
            if 'scale_pred' in return_keys:
                scale_loss = du.all_reduce([scale_loss])[0]

            if 'center_pred' in return_keys:
                center_loss = du.all_reduce([center_loss])[0]
            
            if 'class_pred' in return_keys:
                class_loss = du.all_reduce([class_loss])[0]
            
            if 'img_embed' in return_keys:
                obj_category_triplet_loss, obj_id_triplet_loss = du.all_reduce([obj_category_triplet_loss, obj_id_triplet_loss])

        if du.is_master_proc(num_gpus=args.num_gpus):
            
            if wandb_enabled:
                wandb_dict = {
                    'train/train_loss':total_loss.item(), 
                }
                if 'scale_pred' in return_keys:
                    wandb_dict.update({
                        'train/scale_loss': scale_loss.item(),
                    })

                if 'center_pred' in return_keys:
                    wandb_dict.update({
                        'train/center_loss': center_loss.item(),
                    })
                
                if 'class_pred' in return_keys:
                    wandb_dict.update({
                        'train/class_loss': class_loss.item(),
                    })
                
                if 'img_embed' in return_keys:
                    wandb_dict.update({
                        'train/obj_category_triplet_loss': obj_category_triplet_loss.item(),
                        'train/obj_id_triplet_loss': obj_id_triplet_loss.item(),
                    })
                wandb.log(wandb_dict, step=cnt)

            if cnt % args.training_config.log_every == 0:
                
                logger.info('Train Epoch: {} [iter={} ({:.0f}%)]'.format(epoch, cnt, 100. * batch_idx / len(train_loader)))
                logger.info('\tTotal Loss = {:.6f}'.format(total_loss.item()))
                if 'scale_pred' in return_keys:
                    logger.info('\tscale_loss={:.6f}'.format(scale_loss.item()))

                if 'center_pred' in return_keys:
                    logger.info('\tcenter_loss={:.6f}'.format(center_loss.item()))
                
                if 'class_pred' in return_keys:
                    logger.info('\tclass_loss={:.6f}'.format(class_loss.item()))
                
                if 'img_embed' in return_keys:
                    logger.info('\tobj_category_triplet_loss={:.6f}'.format(obj_category_triplet_loss.item()))
                    logger.info('\tobj_id_triplet_loss={:.6f}'.format(obj_id_triplet_loss.item()))
            
        if du.is_master_proc(num_gpus=args.num_gpus):
            if 'img_embed' in return_keys and cnt % args.training_config.plot_triplet_every == 0:               
                image_tensor = image.cpu().detach()[:,:3,:,:]
                mask_tensor = image.cpu().detach()[:,3:,:,:]
                image_tensor = utrans.denormalize(image_tensor, train_loader.dataset.img_mean, train_loader.dataset.img_std)
                
                mask_L = [
                    ("obj_id", obj_id, triplet_mask_obj_id),
                    ("obj_category", obj_category, triplet_mask_obj_category),
                ]
                for gt_key, gt_value, mask in mask_L:
                    gt_value = gt_value.detach().cpu().numpy()
                    sample_ids = data["sample_id"].numpy().astype(int).astype(str)
                    triplets = torch.stack(torch.where(mask), dim=1)
                    plt_pairs_idx = np.random.choice(len(triplets), args.training_config.triplet_plot_num, replace=False)
                    triplets = triplets[list(plt_pairs_idx)]
                    for triplet in triplets:
                        fig, axs = plt.subplots(1, 3, figsize=(30,20)) 
                        triplet_sample_ids = ['-'.join(sample_ids[idx]) for idx in triplet]
                        for i in range(3):
                            idx_in_batch = triplet[i]
                            gt_value_i = gt_value[idx_in_batch]
                            image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
                            mask_PIL = torchvision.transforms.ToPILImage()(mask_tensor[idx_in_batch])
                            obj_background = PIL.Image.new("RGB", image_PIL.size, 0)
                            masked_image = PIL.Image.composite(image_PIL, obj_background, mask_PIL)
                            axs[i].imshow(np.asarray(masked_image))
                            axs[i].set_title('gt={}'.format(gt_value_i))
                        image_name = '{}_{}-samples={}'.format(epoch, cnt, '_'.join(triplet_sample_ids))
                        if wandb_enabled:
                            final_img = uplot.plt_to_image(fig)
                            log_key = '{}/{}'.format(gt_key, image_name)
                            wandb.log({log_key: wandb.Image(final_img)}, step=cnt)
                        else:
                            image_path = os.path.join(image_dir, "{}_{}.png".format(gt_key, image_name))
                            plt.savefig(image_path)
                        plt.close()
        torch.cuda.empty_cache()            
        cnt += 1
    return cnt


def train(args):
    if du.is_master_proc():
        if args.wandb.enable and args.training_config.train:
            wandb.login()
            wandb.init(project=args.wandb.wandb_project_name, entity=args.wandb.wandb_project_entity, config=args.obj_dict)
    wandb_enabled = args.wandb.enable and not wandb.run is None
    if wandb_enabled:
        wandb_run_name = wandb.run.name 
    else:
        wandb_run_name = uu.get_timestamp()
    
    if du.is_master_proc(num_gpus=args.num_gpus):
        this_experiment_dir, image_dir, model_dir, prediction_dir = uu.create_experiment_dirs(args, wandb_run_name)
        logging.setup_logging(log_to_file=args.log_to_file, experiment_dir=this_experiment_dir)
    else:
        image_dir,model_dir,prediction_dir = None,None,None
    
    if not args.training_config.train and (args.model_config.model_path == '' or args.model_config.model_path is None):
        logger.warning("Not training, but no provided model path")
    model = build_model(args)
    uu.load_model_from(args, model, data_parallel=args.num_gpus>1)
    
    test_dataset = incat_dataset.InCategoryClutterDataset('test', args)
    test_loader = incat_dataloader.InCategoryClutterDataloader(test_dataset, args, shuffle = False)
    if args.testing_config.feature_extract:
        test_meter = FeatureExtractMeter(args)
    else:
        test_meter = TestMeter(args)
    logger.info("Length of test_dataset: {}, Number of batches: {}".format(
        len(test_dataset),
        len(test_loader),
    ))
    
    if args.training_config.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer_config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_config.step, gamma=args.scheduler_config.gamma)
    
        train_dataset = incat_dataset.InCategoryClutterDataset('train', args)
        train_loader = incat_dataloader.InCategoryClutterDataloader(train_dataset, args, shuffle = True)

        logger.info("Length of train_dataset: {}, Number of batches: {}".format(
            len(train_dataset),
            len(train_loader),
        ))
    
        
    cnt = 0
    if args.training_config.train:
        for epoch in range(args.training_config.start_epoch, args.training_config.epochs):
            # test_loader.set_epoch(epoch)
            # test.test(args, test_loader, test_meter, model, epoch, cnt, image_dir, prediction_dir, wandb_enabled)

            train_loader.set_epoch(epoch)
            cnt = train_epoch(args, train_loader, model, optimizer, epoch, cnt, image_dir, wandb_enabled)

            if du.is_master_proc(num_gpus=args.num_gpus):
                if save_this_epoch(args.training_config, epoch):
                    uu.save_model(epoch, model, model_dir)

            if scheduler is not None:
                scheduler.step()
            
            test_loader.set_epoch(epoch)
            test.test(args, test_loader, test_meter, model, epoch, cnt, image_dir, prediction_dir, wandb_enabled)
    
        if du.is_master_proc(num_gpus=args.num_gpus):
            if args.training_config.save_at_end:
                uu.save_model(args.training_config.epochs, model, model_dir)

    test.test(args, test_loader, test_meter, model, args.training_config.epochs, cnt, image_dir, prediction_dir, wandb_enabled)