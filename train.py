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
    assert len(args.model_config.model_return) == len(training_config.gt) == len(training_config.loss_fn) == len(training_config.weight)
    
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        mask_dict = {} # gt_key -> mask used for triplet loss 
        cuda_item = {}
        arguments = []
        for arg_key in args.model_config.model_arguments:
            cuda_item[arg_key] = data[arg_key].cuda(non_blocking=args.cuda_non_blocking)
            arguments += [cuda_item[arg_key]]
        returns = model(arguments)
        
        loss_dict = {}
        for loss_idx, loss_fn_name in enumerate(training_config.loss_fn):
            gt_key = training_config.gt[loss_idx]
            if gt_key in cuda_item:
                gt_val = cuda_item[gt_key]
            else:
                cuda_item[gt_key] = data[gt_key].cuda(non_blocking=args.cuda_non_blocking)
                gt_val = cuda_item[gt_key]
            
            pred_key = training_config.model_return[loss_idx]
            pred_val = returns[pred_key]

            if loss_fn_name == 'triplet_loss':
                triplet_mask, loss_value = loss.batch_all_triplet_loss(
                    labels=gt_val, 
                    embeddings=pred_val, 
                    margin=args.loss.margin, 
                    squared=False,
                )
                mask_dict[gt_key] = triplet_mask
            else:
                loss_func = loss.get_loss_func(loss_fn_name)
                loss_value = loss_func(pred_val, gt_val)
            loss_dict[loss_fn_name] = training_config.weight[loss_idx] * loss_value

        total_loss = None
        for loss_fn_name, loss_value in loss_dict.items():
            if total_loss is None:
                total_loss = loss_value 
            else:
                total_loss += loss_value
        total_loss.backward()
        optimizer.step()

        if args.num_gpus > 1:
            total_loss = du.all_reduce([total_loss])
            for loss_fn_name, loss_value in loss_dict.items():
                loss_dict[loss_fn_name] = du.all_reduce([loss_value])

        if du.is_master_proc(num_gpus=args.num_gpus):
            
            if wandb_enabled:
                wandb_dict = {
                    'train/train_loss':total_loss.item(), 
                }
                for loss_fn_name, loss_value in loss_dict.items():
                    wandb_dict.update({
                        'train/{}'.format(loss_fn_name) : loss_value.item(),
                    })
                wandb.log(wandb_dict, step=cnt)

            if cnt % args.training_config.log_every == 0:
                
                logger.info('\n')
                logger.info('Train Epoch: {} [iter={} ({:.0f}%)]'.format(epoch, cnt, 100. * batch_idx / len(train_loader)))
                logger.info('\tTotal Loss = {:.6f}'.format(total_loss.item()))
                
                for loss_idx, loss_fn_name in enumerate(training_config.loss_fn):
                    gt_key = training_config.gt[loss_idx]
                    pred_key = training_config.model_return[loss_idx]
                    weight = training_config.weight[loss_idx]
                    
                    logger.info(
                        '\tLoss_fn_name={}, Loss_gt_name={}, Loss_pred_key={}, Loss_weight={}, Loss={:.6f}'.format(
                            loss_fn_name, 
                            gt_key,
                            pred_key,
                            weight,
                            loss_dict[loss_fn_name].item(),
                        )
                    )
            
        if du.is_master_proc(num_gpus=args.num_gpus):
            if cnt % args.training_config.plot_triplet_every == 0 and len(mask_dict) > 0:                    
                if 'image' in cuda_item:
                    image = cuda_item['image']
                    image_tensor = image.cpu().detach()[:,:3,:,:]
                    mask_tensor = image.cpu().detach()[:,3:,:,:]
                else:
                    image = cuda_item['image']
                    image_tensor = image[:,:3,:,:]
                    mask_tensor = image[:,3:,:,:]
                image_tensor = utrans.denormalize(image_tensor, train_loader.dataset.img_mean, train_loader.dataset.img_std)
                
                for gt_key, mask in mask_dict.items():
                    if gt_key not in cuda_item:
                        logger.info("WARNING: This mask {} GT value never sent to cuda, giving up plotting.".format(gt_key))
                    gt_value = cuda_item[gt_key].detach().cpu().numpy()
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