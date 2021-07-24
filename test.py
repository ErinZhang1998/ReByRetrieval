import torch
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
import copy
import wandb
import os 

import utils.utils as uu
import losses.loss as loss
import utils.plot_image as uplot
import utils.transforms as utrans
import utils.distributed as du

def test(args, test_loader, test_meter, model, epoch, cnt, image_dir=None, prediction_dir=None, wandb_enabled=False):
    model.eval()

    plot_step = len(test_loader) // args.testing_config.num_gt_image_plot
    plot_batch_idx = np.arange(args.testing_config.num_gt_image_plot) * plot_step
    
    with torch.no_grad():
        
        for batch_idx, data in enumerate(test_loader):
            # print(batch_idx)
            image = data["image"]
            scale_gt = data["scale"]
            pixel_gt = data["center"]
            cat_gt = data["obj_category"]
            id_gt = data["obj_id"]

            sample_id = data["sample_id"]
            area_type = data["area_type"]
            
            # Send model and data to CUDA
            image = image.cuda(non_blocking=args.cuda_non_blocking)
            if args.use_pc:
                pts = data["obj_points"].cuda(non_blocking=args.cuda_non_blocking)
                feats = data["obj_points_features"].cuda(non_blocking=args.cuda_non_blocking)
                img_embed, pose_pred = model([image, pts, feats])
            else:
                img_embed, pose_pred = model([image])
           
            cat_gt = cat_gt.cuda(non_blocking=args.cuda_non_blocking)
            id_gt = id_gt.cuda(non_blocking=args.cuda_non_blocking)

            # Normalize the image embedding
            img_embed -= img_embed.min(1, keepdim=True)[0]
            img_embed /= img_embed.max(1, keepdim=True)[0]
            
            _,c_loss = loss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=args.loss.margin, squared=False) #.cpu()
            _,o_loss = loss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=args.loss.margin, squared=False) #.cpu()  
            
            if args.num_gpus > 1:
                c_loss, o_loss = du.all_reduce(
                    [c_loss, o_loss]
                )

                image, img_embed, pose_pred = du.all_gather(
                    [image, img_embed, pose_pred]
                )

                scale_gt = torch.cat(du.all_gather_unaligned(scale_gt), dim=0)
                pixel_gt = torch.cat(du.all_gather_unaligned(pixel_gt), dim=0)
                sample_id = torch.cat(du.all_gather_unaligned(sample_id), dim=0)
                area_type = torch.cat(du.all_gather_unaligned(area_type), dim=0)
            
            pixel_pred = pose_pred[:,:2]
            scale_pred = pose_pred[:,2:]          
            
            iter_data = {
                'loss_cat': c_loss.item(),
                'loss_obj': o_loss.item(),
                'image': image.detach().cpu(),
                'embeds': img_embed.detach().cpu(),
                'scale_pred': scale_pred.detach().cpu(),
                'scale_gt': scale_gt.detach().cpu(),
                'pixel_pred': pixel_pred.detach().cpu(),
                'pixel_gt': pixel_gt.detach().cpu(),
                'sample_id': sample_id.detach().cpu(),
                'area_type': area_type.detach().cpu(),
            }
            plot_iter = batch_idx in plot_batch_idx and batch_idx != len(test_loader)-1
            test_meter.log_iter_stats(iter_data, cnt, image_dir, wandb_enabled=wandb_enabled, plot=plot_iter)
            torch.cuda.empty_cache()
    
    if du.is_master_proc(num_gpus=args.num_gpus):
        test_meter.finalize_metrics(epoch, cnt, prediction_dir)
    test_meter.reset()