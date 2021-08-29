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
            
            image = data['image'].cuda(non_blocking=args.cuda_non_blocking)
            return_keys, return_val = model([image])

            sample_id = data['sample_id']
            scale = data['scale']
            obj_category = data['obj_category']
            obj_id = data['obj_id']
            shapenet_model_id = data['shapenet_model_id']
            
            scale_pred = return_val[return_keys.index('scale_pred')]
            if 'class_pred' in return_keys:
                if args.num_gpus > 1:
                    test_meter.num_classes = model.module.num_classes
                else:
                    test_meter.num_classes = model.num_classes
                
                class_pred = return_val[return_keys.index('class_pred')]
                # if args.model_config.class_type == 'shapenet_model_id':
                #     shapenet_model_id = data['shapenet_model_id']
            
            if 'center_pred' in return_keys:
                center = data['center']
                center_pred = return_val[return_keys.index('center_pred')]
            
            if 'img_embed' in return_keys:
                img_embed = return_val[return_keys.index('img_embed')]
                obj_category = obj_category.cuda()
                obj_id = obj_id.cuda()
                
                if args.testing_config.calculate_triplet_loss:
                
                    obj_category_mask, obj_category_loss = loss.batch_all_triplet_loss(
                        labels=obj_category, 
                        embeddings=img_embed, 
                        margin=args.loss.margin, 
                        squared=False,
                    )
                    obj_category_mask = obj_category_mask.float()

                    
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
                shapenet_model_id = torch.cat(du.all_gather_unaligned(shapenet_model_id), dim=0)
                
                image, scale_pred = du.all_gather(
                    [image, scale_pred]
                )
                if 'class_pred' in return_keys:
                    class_pred = du.all_gather([class_pred])[0]
                    # if args.model_config.class_type == 'shapenet_model_id':
                        # shapenet_model_id = torch.cat(du.all_gather_unaligned(shapenet_model_id), dim=0)

                if 'center_pred' in return_keys:
                    center_pred = du.all_gather([center_pred])[0]
                    center = torch.cat(du.all_gather_unaligned(center), dim=0)
                
                if 'img_embed' in return_keys:

                    if args.testing_config.calculate_triplet_loss:
                        img_embed, obj_category_mask, obj_id_mask, obj_category, obj_id = du.all_gather(
                            [img_embed, obj_category_mask, obj_id_mask, obj_category, obj_id]
                        )
                        obj_category_loss, obj_id_loss = du.all_reduce([obj_category_loss, obj_id_loss])
                    else:
                        img_embed, obj_category, obj_id = du.all_gather(
                            [img_embed, obj_category, obj_id]
                        )
                else:
                    obj_category = torch.cat(du.all_gather_unaligned(obj_category), dim=0)
                    obj_id = torch.cat(du.all_gather_unaligned(obj_id), dim=0)
                


            if du.is_master_proc(num_gpus=args.num_gpus):
                iter_data = {
                    'sample_id' : sample_id.detach().cpu(),
                    'shapenet_model_id' : shapenet_model_id.detach().cpu(),
                    'image' : image.detach().cpu(),
                    'scale' : scale.detach().cpu(),
                    'scale_pred' : scale_pred.detach().cpu(),
                    'sample_id' : sample_id.detach().cpu(),
                    'obj_category' : obj_category.detach().cpu(),
                    'obj_id' : obj_id.detach().cpu(),
                }
                
                if 'class_pred' in return_keys:
                    iter_data.update({
                        'class_pred' : class_pred.detach().cpu(),
                    })
                    # if args.model_config.class_type == 'shapenet_model_id':
                    #     iter_data.update({
                    #         'shapenet_model_id' : shapenet_model_id.detach().cpu(),
                    #     })
                
                if 'center_pred' in return_keys:
                    iter_data.update({
                        'center' : center.detach().cpu(),
                        'center_pred' : center_pred.detach().cpu(),
                    })
                
                if 'img_embed' in return_keys:
                    if args.testing_config.calculate_triplet_loss:
                        num_triplet_obj_category = torch.sum(obj_category_mask).item()
                        num_triplet_obj_id = torch.sum(obj_id_mask).item()
                        iter_data.update({
                            'img_embed' : img_embed.detach().cpu(),
                            'contrastive_obj_category' : (obj_category_loss.item() * num_triplet_obj_category, num_triplet_obj_category),
                            'contrastive_obj_id' : (obj_id_loss.item() * num_triplet_obj_id, num_triplet_obj_id),
                        })
                    else:
                        iter_data.update({
                            'img_embed' : img_embed.detach().cpu(),
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