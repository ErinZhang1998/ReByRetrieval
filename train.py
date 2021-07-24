from __future__ import print_function

import torch
import numpy as np
import os
from collections import OrderedDict
import torchvision
import PIL
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils.plot_image as uplot 
import utils.transforms as utrans
import utils.utils as uu
import utils.distributed as du
from utils.meters import TestMeter

import losses.loss as loss

from models.build import build_model
import incat_dataset
import incat_dataloader
import test

def save_this_epoch(args, epoch):

    if args.save_freq < 0:
        return False 
    return epoch % args.save_freq == 0

def remove_module_key_transform(key):
    parts = key.split(".")
    if parts[0] == 'module':
        return ".".join(parts[1:])
    return key

def rename_state_dict_keys(ckp_path, key_transformation):
    state_dict = torch.load(ckp_path)['model_state_dict']
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    return new_state_dict

def load_model_from(args, model, data_parallel=False):
    ms = model.module if data_parallel else model
    if args.model_config.model_path is not None:
        print("=> Loading model file from: ", args.model_config.model_path)
        ckp_path = os.path.join(args.model_config.model_path)
        checkpoint = torch.load(ckp_path)
        try:
            ms.load_state_dict(checkpoint['model_state_dict'])
        except:
            state_dict = rename_state_dict_keys(ckp_path, remove_module_key_transform)
            ms.load_state_dict(state_dict)

def save_model(epoch, model, model_dir):
    model_path = os.path.join(model_dir, '{}.pth'.format(epoch))
    try:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, model_path)
    except:
        print("ERROR: Cannot save model at", model_path)

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

        if args.use_pc:
            pts = data["obj_points"].cuda(non_blocking=args.cuda_non_blocking)
            feats = data["obj_points_features"].cuda(non_blocking=args.cuda_non_blocking)
            
            img_embed, pose_pred = model([image, pts, feats])
        else:
            img_embed, pose_pred = model([image])
        # Position prediction
        if args.predict_center: 
            scale_start_idx = 2
            pixel_pred = pose_pred[:,:scale_start_idx]
        else:
            scale_start_idx = 0
        scale_pred = pose_pred[:,scale_start_idx:]
        # Normalize embedding
        img_embed -= img_embed.min(1, keepdim=True)[0]
        img_embed /= img_embed.max(1, keepdim=True)[0]
        '''
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        '''
        mask_cat, loss_cat = loss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=args.loss.margin, squared=False)
        mask_id, loss_obj = loss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=args.loss.margin, squared=False)
        
        loss_fun = loss.get_loss_func(args.training_config.loss_used)(reduction="mean")
        loss_scale = loss_fun(scale_pred, scale_gt)
        loss_cat_w = args.loss.lambda_cat * loss_cat
        loss_obj_w = args.loss.lambda_obj * loss_obj
        loss_scale_w = args.loss.lambda_scale * loss_scale

        if args.predict_center: 
            loss_pixel = loss_fun(pixel_pred, pixel_gt)
            loss_pixel_w = args.loss.lambda_pixel * loss_pixel
            total_loss = loss_cat_w + loss_obj_w + loss_scale_w + loss_pixel_w
        else:
            total_loss = loss_cat_w + loss_obj_w + loss_scale_w
        total_loss.backward()
        
        optimizer.step()

        if args.num_gpus > 1:
            if args.predict_center: 
                total_loss, loss_cat_w, loss_obj_w, loss_scale_w, loss_pixel_w = du.all_reduce(
                    [total_loss, loss_cat_w, loss_obj_w, loss_scale_w, loss_pixel_w]
                )
            else:
                total_loss, loss_cat_w, loss_obj_w, loss_scale_w = du.all_reduce(
                    [total_loss, loss_cat_w, loss_obj_w, loss_scale_w]
                )

        if du.is_master_proc(num_gpus=args.num_gpus):
            
            if wandb_enabled:
                wandb_dict = {
                    'train/train_loss':total_loss.item(), 
                    'train/train_loss_cat': loss_cat_w.item(), 
                    'train/train_loss_obj': loss_obj_w.item(), 
                    'train/train_loss_scale': loss_scale_w.item(), 
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }
                if args.predict_center: 
                    wandb_dict.update({'train/train_loss_pixel': loss_pixel_w.item()})
                wandb.log(wandb_dict, step=cnt)

            if cnt % args.training_config.log_every == 0:
                if args.predict_center: 
                    loss_pixel_w_item = loss_pixel_w.item()
                else:
                    loss_pixel_w_item = -1
                print('Train Epoch: {} [{} ({:.0f}%)]\tTotal Loss={:.6f}, Triplet_Loss_Category ({}) = {:.6f}, Triplet_Loss_Object ({}) = {:.6f}, Object_Scale_Loss ({}) = {:.6f}, Object_2D_Center_Loss ({}) = {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), total_loss.item(), \
                        args.loss.lambda_cat, loss_cat_w.item(), \
                        args.loss.lambda_obj, loss_obj_w.item(), \
                        args.loss.lambda_scale, loss_scale_w.item(), \
                        args.loss.lambda_pixel, loss_pixel_w_item))

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
    # import pdb; pdb.set_trace()
    model = build_model(args)
    load_model_from(args, model, data_parallel=args.num_gpus>1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_config.step, gamma=args.scheduler_config.gamma)
    
    train_dataset = incat_dataset.InCategoryClutterDataset('train', args)
    train_loader = incat_dataloader.InCategoryClutterDataloader(train_dataset, args, shuffle = True)

    test_dataset = incat_dataset.InCategoryClutterDataset('test', args)
    test_loader = incat_dataloader.InCategoryClutterDataloader(test_dataset, args, shuffle = False)
    
    wandb_enabled = args.wandb.enable and not wandb.run is None
    if wandb_enabled:
        wandb_run_name = wandb.run.name 
    else:
        wandb_run_name = uu.get_timestamp()
    
    if du.is_master_proc(num_gpus=args.num_gpus):
        if args.training_config.experiment_save_dir is None:
            experiment_save_dir_default = args.training_config.experiment_save_dir_default
            uu.create_dir(experiment_save_dir_default)
            this_experiment_dir = os.path.join(experiment_save_dir_default, wandb_run_name)
            uu.create_dir(this_experiment_dir)

            model_dir = os.path.join(this_experiment_dir, "models")
            uu.create_dir(model_dir)
            
            image_dir = os.path.join(this_experiment_dir, "images")
            uu.create_dir(image_dir)

            prediction_dir = os.path.join(this_experiment_dir, "predictions")
            uu.create_dir(prediction_dir)
        else:
            this_experiment_dir = args.training_config.experiment_save_dir
            model_dir = os.path.join(this_experiment_dir, "models")            
            image_dir = os.path.join(this_experiment_dir, "images")
            prediction_dir = os.path.join(this_experiment_dir, "predictions")
    else:
        image_dir,model_dir,prediction_dir = None,None,None

    test_meter = TestMeter(args)
        
    cnt = 0
    for epoch in range(args.training_config.start_epoch, args.training_config.epochs):
        # test_loader.set_epoch(epoch)
        # test.test(args, test_loader, test_meter, model, epoch, cnt, image_dir, prediction_dir, wandb_enabled)

        train_loader.set_epoch(epoch)
        cnt = train_epoch(args, train_loader, model, optimizer, epoch, cnt, image_dir, wandb_enabled)

        if du.is_master_proc(num_gpus=args.num_gpus):
            if save_this_epoch(args.training_config, epoch):
                save_model(epoch, model, model_dir)

        if scheduler is not None:
            scheduler.step()
        
        test_loader.set_epoch(epoch)
        test.test(args, test_loader, test_meter, model, epoch, cnt, image_dir, prediction_dir, wandb_enabled)
    
    if du.is_master_proc(num_gpus=args.num_gpus):
        if args.training_config.save_at_end:
            save_model(args.training_config.epochs, model, model_dir)

    # test.test(args, test_loader, test_meter, model, args.training_config.epochs, cnt, image_dir, prediction_dir, wandb_enabled)