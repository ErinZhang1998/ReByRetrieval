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
            contrastive_loss_dict = {}
            mask_dict = {}
            cuda_item = {}
            arguments = []
            for arg_key in args.model_config.model_arguments:
                cuda_item[arg_key] = data[arg_key].cuda(non_blocking=args.cuda_non_blocking)
                arguments += [cuda_item[arg_key]]
            returns = model(arguments)
            gt_dict = {}
            sample_id = data["sample_id"]
            for key in args.training_config.gt:
                gt_dict[key] = data[key]
            
            #### Contrastive loss need to use GPU_CUDA
            for loss_idx, loss_fn_name in enumerate(args.training_config.loss_fn):
                if loss_fn_name != 'triplet_loss':
                    continue
                gt_key = args.training_config.gt[loss_idx]
                if gt_key in cuda_item:
                    gt_val = cuda_item[gt_key]
                else:
                    cuda_item[gt_key] = data[gt_key].cuda(non_blocking=args.cuda_non_blocking)
                    gt_val = cuda_item[gt_key]
                
                pred_key = args.model_config.model_return[loss_idx]
                pred_val = returns[pred_key]
                
                triplet_mask, loss_value = loss.batch_all_triplet_loss(
                    labels=gt_val, 
                    embeddings=pred_val, 
                    margin=args.loss.margin, 
                    squared=False,
                )
                contrastive_loss_dict[gt_key] = loss_value * torch.sum(triplet_mask).item()
                mask_dict[gt_key] = triplet_mask
            ####
            if args.num_gpus > 1:
                for key, loss_value in contrastive_loss_dict.items():
                    new_item = du.all_reduce([loss_value])[0]
                    contrastive_loss_dict[key] = new_item
                print(contrastive_loss_dict.keys())
                for key, mask in mask_dict.items():
                    new_item = du.all_gather([mask.float()])[0]
                    mask_dict[key] = new_item
                print(mask_dict.keys())
                for key, cuda_tensor in cuda_item.items():
                    new_item = du.all_gather([cuda_tensor])[0]
                    cuda_item[key] = new_item
                print(cuda_item.keys())
                for key, return_tensor in returns.items():
                    new_item = du.all_gather([return_tensor])[0]
                    returns[key] = new_item
                print(returns.keys())
                print(cuda_item['image'])
                print(returns['img_embed'])
                
                for key in args.training_config.gt:
                    new_item = torch.cat(du.all_gather_unaligned(data[key]), dim=0)
                    gt_dict[key] = new_item

                sample_id = torch.cat(du.all_gather_unaligned(sample_id), dim=0)
                logger.info("hi")

            if du.is_master_proc(num_gpus=args.num_gpus):
                logger.info("hi")
                iter_data = {}
                print(cuda_item['image'])
                for key in cuda_item.keys():
                    print(cuda_item[key])
                    new_item = cuda_item[key].detach().cpu()
                    iter_data[key] = new_item

                for key, return_tensor in returns.items():
                    new_item = return_tensor.detach().cpu()
                    iter_data[key] = new_item
                
                for key, gt_value in args.training_config.gt.items():
                    new_item = gt_value.detach().cpu()
                    iter_data[key] = new_item
                
                for key, loss_value in contrastive_loss_dict.items():
                    new_item_1 = loss_value.item()
                    new_item_2 = torch.sum(mask_dict[key]).item()

                    iter_data['contrastive_{}'.format(key)] = (new_item_1, new_item_2)
                
                logger.info("hi")
                new_sample_id = sample_id.detach().cpu()
                iter_data['sample_id'] = new_sample_id
                
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