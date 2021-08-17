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
            if args.num_gpus > 1:
                for key, cuda_tensor in cuda_item.items():
                    cuda_item[key] = du.all_gather([cuda_tensor])
                for key, return_tensor in returns.items():
                    returns[key] = du.all_gather([return_tensor])
                for key in args.training_config.gt:
                    gt_dict[key] = torch.cat(du.all_gather_unaligned(data[key]), dim=0)
                
                sample_id = torch.cat(du.all_gather_unaligned(sample_id), dim=0)
            
            iter_data = {}
            for key, cuda_tensor in cuda_item.items():
                iter_data[key] = cuda_tensor.detach().cpu()
            for key, return_tensor in returns.items():
                iter_data[key] = return_tensor.detach().cpu()
            for key in args.training_config.gt:
                iter_data[key] = gt_dict[key]
            iter_data['sample_id'] = sample_id.detach().cpu()
            
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