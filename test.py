import torch
import numpy as np

import losses.loss as loss
import utils.distributed as du
import utils.logging as logging 

logger = logging.get_logger(__name__)

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

            if args.model_config.classification:
                loss_classification = torch.nn.CrossEntropyLoss()(class_pred, classification_gt)
            else: 
                # Normalize the image embedding
                img_embed -= img_embed.min(1, keepdim=True)[0]
                img_embed /= img_embed.max(1, keepdim=True)[0]
                _,c_loss = loss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=args.loss.margin, squared=False)
                _,o_loss = loss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=args.loss.margin, squared=False) 

            if args.num_gpus > 1:
                image, pose_pred = du.all_gather([image, pose_pred])
                scale_gt = torch.cat(du.all_gather_unaligned(scale_gt), dim=0)
                sample_id = torch.cat(du.all_gather_unaligned(sample_id), dim=0)

                if args.model_config.predict_center: 
                    pixel_gt = torch.cat(du.all_gather_unaligned(pixel_gt), dim=0)
                
                if args.model_config.classification:
                    class_pred = du.all_gather([class_pred])
                    loss_classification = du.all_reduce([loss_classification])
                else: 
                    img_embed = du.all_gather([img_embed])
                    c_loss, o_loss = du.all_reduce([c_loss, o_loss])
                    cat_gt = torch.cat(du.all_gather_unaligned(cat_gt), dim=0)
                    id_gt = torch.cat(du.all_gather_unaligned(id_gt), dim=0)
                
                if not args.blender_proc:
                    area_type = torch.cat(du.all_gather_unaligned(area_type), dim=0)
            
            if args.model_config.predict_center: 
                scale_start_idx = 2
                pixel_pred = pose_pred[:,:scale_start_idx]
            else:
                scale_start_idx = 0
            scale_pred = pose_pred[:,scale_start_idx:] 
            
            iter_data = {
                'image': image.detach().cpu(),
                'scale_pred': scale_pred.detach().cpu(),
                'scale_gt': scale_gt.detach().cpu(),
                'sample_id': sample_id.detach().cpu(),
            }

            if args.model_config.predict_center: 
                iter_data.update({
                    'pixel_pred': pixel_pred.detach().cpu(),
                    'pixel_gt': pixel_gt.detach().cpu(),
                })
            
            if args.model_config.classification:
                iter_data.update({
                    'class_pred': class_pred.detach().cpu(),
                    'loss_classification': loss_classification.item(),
                })
            else: 
                iter_data.update({
                    'loss_cat': c_loss.item(),
                    'loss_obj': o_loss.item(),
                    'embeds': img_embed.detach().cpu(),
                    'cat_gt' : cat_gt.detach().cpu(),
                    'id_gt' : id_gt.detach().cpu(),
                })
            if not args.blender_proc:
                iter_data.update({
                    'area_type': area_type.detach().cpu(),
                })
            
            plot_iter = batch_idx in plot_batch_idx and batch_idx != len(test_loader)-1
            test_meter.log_iter_stats(iter_data, cnt, batch_idx, image_dir, wandb_enabled=wandb_enabled, plot=plot_iter)
            torch.cuda.empty_cache()
    
    if du.is_master_proc(num_gpus=args.num_gpus):
        test_meter.finalize_metrics(epoch, cnt, prediction_dir)
    test_meter.reset()