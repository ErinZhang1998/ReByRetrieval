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
            sample_id = data['sample_id']
            image = data['image'].cuda(non_blocking=args.cuda_non_blocking)
            returns = model([image])

            scale = data['scale']
            scale_pred = returns['scale_pred']
            if 'class_pred' in returns:
                class_pred = returns['class_pred']
                if args.model_config.class_type == 'shapenet_model_id':
                    shapenet_model_id = data['shapenet_model_id']
            
            if 'center_pred' in returns:
                center = data['center']
                center_pred = returns['center_pred']
            
            if 'triplet_loss' in args.training_config.loss_fn:
                img_embed = returns['img_embed']
                obj_category = data['obj_category'].cuda(non_blocking=args.cuda_non_blocking)
                obj_category_mask, obj_category_loss = loss.batch_all_triplet_loss(
                    labels=obj_category, 
                    embeddings=img_embed, 
                    margin=args.loss.margin, 
                    squared=False,
                )
                obj_category_mask = obj_category_mask.float()

                obj_id = data['obj_id'].cuda(non_blocking=args.cuda_non_blocking)
                obj_id_mask, obj_id_loss = loss.batch_all_triplet_loss(
                    labels=obj_id, 
                    embeddings=img_embed, 
                    margin=args.loss.margin, 
                    squared=False,
                )
                obj_id_mask = obj_id_mask.float()
            
            if args.num_gpus > 1:
                sample_id = torch.cat(du.all_gather_unaligned(sample_id), dim=0)
                scale = torch.cat(du.all_gather_unaligned(scale), dim=0)
                
                image, scale_pred = du.all_gather(
                    [image, scale_pred]
                )
                if 'class_pred' in returns:
                    class_pred = du.all_gather([class_pred])[0]
                    if args.model_config.class_type == 'shapenet_model_id':
                        shapenet_model_id = torch.cat(du.all_gather_unaligned(shapenet_model_id), dim=0)

                if 'center_pred' in returns:
                    center_pred = du.all_gather([center_pred])[0]
                    center = torch.cat(du.all_gather_unaligned(center), dim=0)
                
                if 'triplet_loss' in args.training_config.loss_fn:
                    img_embed, obj_category_mask, obj_id_mask = du.all_gather(
                        [img_embed, obj_category_mask, obj_id_mask]
                    )
                    obj_category = torch.cat(du.all_gather_unaligned(obj_category), dim=0)
                    obj_id = torch.cat(du.all_gather_unaligned(obj_id), dim=0)
                    obj_category_loss, obj_id_loss = du.all_reduce([obj_category_loss, obj_id_loss])

            if du.is_master_proc(num_gpus=args.num_gpus):
                iter_data = {
                    'sample_id' : sample_id.detach().cpu(),
                    'image' : image.detach().cpu(),
                    'scale' : scale.detach().cpu(),
                    'scale_pred' : scale_pred.detach().cpu(),
                    'sample_id' : sample_id.detach().cpu(),
                }
                
                if 'class_pred' in returns:
                    iter_data.update({
                        'class_pred' : class_pred.detach().cpu(),
                    })
                    if args.model_config.class_type == 'shapenet_model_id':
                        iter_data.update({
                            'shapenet_model_id' : class_pred.detach().cpu(),
                        })
                
                if 'center_pred' in returns:
                    iter_data.update({
                        'center' : center.detach().cpu(),
                        'center_pred' : center_pred.detach().cpu(),
                    })
                
                if 'triplet_loss' in args.training_config.loss_fn:
                    iter_data.update({
                        'img_embed' : img_embed.detach().cpu(),
                        'obj_category' : obj_category.detach().cpu(),
                        'obj_id' : obj_id.detach().cpu(),
                        'contrastive_obj_category' : (obj_category_loss.item(), torch.sum(obj_category_mask).item()),
                        'contrastive_obj_id' : (obj_id_loss.item(), torch.sum(obj_id_mask).item()),
                    })
                
                plot_iter = batch_idx in plot_batch_idx and batch_idx != len(test_loader)-1
                test_meter.log_iter_stats(
                    iter_data, 
                    cnt, 
                    batch_idx, 
                    image_dir, 
                    wandb_enabled=wandb_enabled, 
                    plot=plot_iter
                )
                
            torch.cuda.empty_cache()
    
    if du.is_master_proc(num_gpus=args.num_gpus):
        test_meter.finalize_metrics(epoch, cnt, prediction_dir)
    test_meter.reset()