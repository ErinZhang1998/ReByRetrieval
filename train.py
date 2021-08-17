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
    
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()

        image = data["image"]
        scale_gt = data["scale"]
        pixel_gt = data["center"]
        cat_gt = data["obj_category"]
        id_gt = data["obj_id"]
        sample_id_int = data["sample_id"]
        
        # Send model and data to CUDA
        image = image.cuda(non_blocking=args.cuda_non_blocking)
        scale_gt = scale_gt.cuda(non_blocking=args.cuda_non_blocking)
        pixel_gt = pixel_gt.cuda(non_blocking=args.cuda_non_blocking)
        cat_gt = cat_gt.cuda(non_blocking=args.cuda_non_blocking)
        id_gt = id_gt.cuda(non_blocking=args.cuda_non_blocking)

        if args.model_config.classification:
            if args.model_config.class_type == 'shapenet_model':
                classification_gt = data["shapenet_model_id"].cuda(non_blocking=args.cuda_non_blocking)
            elif args.model_config.class_type == 'shapenet_category':
                classification_gt = cat_gt
            else:
                classification_gt = id_gt
        
        if args.use_pc:
            pts = data["obj_points"].cuda(non_blocking=args.cuda_non_blocking)
            feats = data["obj_points_features"].cuda(non_blocking=args.cuda_non_blocking)
            img_embed, pose_pred = model([image, pts, feats])
        else:
            if args.model_config.classification:
                class_pred, pose_pred = model([image])
            else:
                img_embed, pose_pred = model([image])
        
        # Position prediction
        if args.model_config.predict_center: 
            scale_start_idx = 2
            pixel_pred = pose_pred[:,:scale_start_idx]
        else:
            scale_start_idx = 0
        scale_pred = pose_pred[:,scale_start_idx:]
        loss_fun = loss.get_loss_func(args.training_config.loss_used)(reduction="mean")
        loss_scale = loss_fun(scale_pred, scale_gt)
        loss_scale_w = args.loss.lambda_scale * loss_scale

        if args.model_config.predict_center: 
            loss_pixel = loss_fun(pixel_pred, pixel_gt)
            loss_pixel_w = args.loss.lambda_pixel * loss_pixel

        if args.model_config.classification:
            loss_classification = torch.nn.CrossEntropyLoss()(class_pred, classification_gt)
            loss_classification_w = args.loss.lambda_classification * loss_classification
        else:
            # Normalize embedding
            img_embed -= img_embed.min(1, keepdim=True)[0]
            img_embed /= img_embed.max(1, keepdim=True)[0]
            mask_cat, loss_cat = loss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=args.loss.margin, squared=False)
            mask_id, loss_obj = loss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=args.loss.margin, squared=False)
            loss_cat_w = args.loss.lambda_cat * loss_cat
            loss_obj_w = args.loss.lambda_obj * loss_obj
            
        total_loss = loss_scale_w
        if args.model_config.predict_center: 
            total_loss += loss_pixel_w
        if args.model_config.classification:
            total_loss += loss_classification_w
        else:
            total_loss += loss_cat_w 
            total_loss += loss_obj_w
        total_loss.backward()
        
        optimizer.step()

        if args.num_gpus > 1:
            total_loss, loss_scale_w = du.all_reduce(
                [total_loss, loss_scale_w]
            )
            if args.model_config.predict_center: 
                loss_pixel_w = du.all_reduce([loss_pixel_w])
            
            if args.model_config.classification:
                loss_classification_w = du.all_reduce([loss_classification_w])
            else:
                loss_cat_w = du.all_reduce([loss_cat_w])
                loss_obj_w = du.all_reduce([loss_obj_w])

        if du.is_master_proc(num_gpus=args.num_gpus):
            
            if wandb_enabled:
                wandb_dict = {
                    'train/train_loss':total_loss.item(), 
                    'train/train_loss_scale': loss_scale_w.item(), 
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                if args.model_config.predict_center: 
                    wandb_dict.update({'train/train_loss_pixel': loss_pixel_w.item()})
                
                if args.model_config.classification:
                    wandb_dict.update({
                        'train/train_loss_classification_{}'.format(args.model_config.class_type) : loss_classification_w.item(),
                    })
                else:
                    wandb_dict.update({
                        'train/train_loss_cat': loss_cat_w.item(),
                        'train/train_loss_obj': loss_obj_w.item()
                    })
                wandb.log(wandb_dict, step=cnt)

            if cnt % args.training_config.log_every == 0:
                
                logger.info('\n')
                logger.info('Train Epoch: {} [iter={} ({:.0f}%)]'.format(epoch, cnt, 100. * batch_idx / len(train_loader)))
                logger.info('\tTotal Loss = {:.6f}'.format(total_loss.item()))
                logger.info('\tObject_Scale_Loss ({}) = {:.6f}'.format(args.loss.lambda_scale, loss_scale_w.item()))

                if args.model_config.predict_center: 
                    logger.info('\tObject_2D_Center_Loss ({}) = {:.6f}'.format(
                        args.loss.lambda_pixel, 
                        loss_pixel_w.item()
                        )
                    )
                
                if args.model_config.classification:
                    logger.info('\Classification Loss ({}) ({}) = {:.6f}'.format(
                        args.loss.lambda_classification, 
                        loss_classification_w.item()
                        )
                    )
                else:
                    logger.info('\tTriplet_Loss_Category ({}) = {:.6f}'.format(args.loss.lambda_cat, loss_cat_w.item()))
                    logger.info('\tTriplet_Loss_Object ({}) = {:.6f}'.format(args.loss.lambda_obj, loss_obj_w.item()))

        if du.is_master_proc(num_gpus=args.num_gpus):
            if cnt % args.training_config.plot_triplet_every == 0:
                image_tensor = image.cpu().detach()[:,:3,:,:]
                image_tensor = utrans.denormalize(image_tensor, train_loader.dataset.img_mean, train_loader.dataset.img_std)
                mask_tensor = image.cpu().detach()[:,3:,:,:]

                cat_gt_np = cat_gt.detach().cpu().numpy()
                id_gt_np = id_gt.detach().cpu().numpy()
                sample_id_int_np = sample_id_int.detach().cpu().numpy().astype(int).astype(str)

                for mask,mask_name in [(mask_cat, "mask_cat"), (mask_id, "mask_id")]:
                    triplets = torch.stack(torch.where(mask), dim=1)
                    plt_pairs_idx = np.random.choice(len(triplets), args.training_config.triplet_plot_num, replace=False)
                    triplets = triplets[list(plt_pairs_idx)]
                    
                    for triplet in triplets:
                        fig, axs = plt.subplots(1, 3, figsize=(30,20))  
                        sample_ids = ['-'.join(sample_id_int_np[idx]) for idx in triplet]
                        for i in range(3):
                            idx_in_batch = triplet[i]
                            obj_cat, obj_id = cat_gt_np[idx_in_batch], id_gt_np[idx_in_batch]
                            image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
                            mask_PIL = torchvision.transforms.ToPILImage()(mask_tensor[idx_in_batch])
                            obj_background = PIL.Image.new("RGB", image_PIL.size, 0)
                            masked_image = PIL.Image.composite(image_PIL, obj_background, mask_PIL)
                            axs[i].imshow(np.asarray(masked_image))
                            axs[i].set_title('{}_{}'.format(obj_cat, obj_id))
                        
                        image_name = '{}_{}_{}'.format(epoch, cnt, '_'.join(sample_ids))
                        if wandb_enabled:
                            final_img = uplot.plt_to_image(fig)
                            log_key = '{}/{}'.format(mask_name, image_name)
                            wandb.log({log_key: wandb.Image(final_img)}, step=cnt)
                        else:
                            image_path = os.path.join(image_dir, "{}_{}.png".format(mask_name, image_name))
                            plt.savefig(image_path)
                        plt.close()
            
        torch.cuda.empty_cache()

        # Validation iteration
        # if cnt % args.training_config.val_every == 0:
        #     test.test(args, test_loader, test_meter, model, epoch, cnt, image_dir, prediction_dir, wandb_enabled=False)
        #     model.train()
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