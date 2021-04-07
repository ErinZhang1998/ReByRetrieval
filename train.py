from __future__ import print_function

import torch
import numpy as np
import os
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils.plot_image as uplot 
import utils.utils as uu
import test
import loss.triplet_loss as triploss

def save_this_epoch(args, epoch):

    if args.save_freq < 0:
        return False 
    return epoch % args.save_freq == 0

def save_model(model_save_dir, epoch, wandb_run_name, model):
    model_path_all = os.path.join(model_save_dir, 'models')
    if not os.path.exists(model_path_all):
        os.mkdir(model_path_all)
    
    model_dir = os.path.join(model_save_dir, 'models', '{}'.format(wandb_run_name))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 

    model_path = os.path.join(model_dir, '{}.pth'.format(epoch))
    try:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, model_path)
    except:
        print("ERROR: Cannot save model at", model_path)

class Trainer(object):
    def __init__(self, all_args, model, train_loader, test_loader, optimizer, scheduler, device):

        self.all_args = all_args
        self.args = all_args.training_config 
        self.loss_args = all_args.loss 
        self.test_args = all_args.testing_config
        self.model = model 

        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.device = device
        self.model.train()

        if self.args.loss_used == "l1":
            self.criterion_scale = torch.nn.L1Loss()
            self.criterion_pixel = torch.nn.L1Loss()
        else:
            self.criterion_scale = torch.nn.MSELoss()
            self.criterion_pixel = torch.nn.MSELoss()

        self.cnt = -1

        if wandb.run is None:
            self.wandb_run_name = uu.get_timestamp()
        else:
            self.wandb_run_name = wandb.run.name
        self.load_model_from()
    
    def load_model_from(self):
        if self.all_args.model_config.model_path is not None:
            print("=> Loading model file from: ", self.all_args.model_config.model_path)
            model_path = os.path.join(self.all_args.model_config.model_path)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def train(self, test_only = False):
        start_epoch = self.args.start_epoch
        
        if not test_only: 
            self.cnt = 0
            for epoch in range(start_epoch, self.args.epochs):
                self.train_epoch(epoch)
        
            if self.args.save_at_end:
                save_model(self.args.model_save_dir, self.args.epochs, self.wandb_run_name, self.model)
        

        l1,l2,l3,l4 = test.eval_dataset(self.args.epochs, self.cnt, self.model, self.device, self.test_loader, self.loss_args, self.test_args, self.args.loss_used, True)
        print('Validate End: \tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(l1,l2,l3,l4))


    def train_epoch(self, epoch):

        hist_loss_pixel = 0.0
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            image = data[0]
            scale_gt = data[1]
            pixel_gt = data[2]
            cat_gt = data[3]
            id_gt = data[4]
            dataset_indices = data[5]  
            
            self.model = self.model.to(self.device)

            image = image.to(self.device)
            scale_gt = scale_gt.to(self.device)
            pixel_gt = pixel_gt.to(self.device)
            cat_gt = cat_gt.to(self.device)
            id_gt = id_gt.to(self.device)
            
            img_embed, pose_pred = self.model(image)
            scale_pred = pose_pred[:,:1]
            pixel_pred = pose_pred[:,1:]
            img_embed -= img_embed.min(1, keepdim=True)[0]
            img_embed /= img_embed.max(1, keepdim=True)[0]

            mask_cat, loss_cat = triploss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=self.loss_args.margin, squared=False)
            mask_id, loss_obj = triploss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=self.loss_args.margin, squared=False)
            loss_scale = self.criterion_scale(scale_pred, scale_gt)
            loss_pixel = self.criterion_pixel(pixel_pred, pixel_gt)

            loss_cat_w = self.loss_args.lambda_cat * loss_cat
            loss_obj_w = self.loss_args.lambda_obj * loss_obj
            loss_scale_w = self.loss_args.lambda_scale * loss_scale
            loss_pixel_w = self.loss_args.lambda_pixel * loss_pixel

            loss = loss_cat_w + loss_obj_w + loss_scale_w + loss_pixel_w
            loss.backward()
            
            self.optimizer.step()

            if epoch <= self.args.loss_anormaly_detech_epoch:
                if batch_idx > 0 and loss_pixel_w.item() >= (self.args.loss_anormaly_scale - 1) * hist_loss_pixel:
                    print("Spike in training batch: ", batch_idx, loss_pixel_w.item())
                    dataset_indices_np = dataset_indices.cpu().numpy().astype(int).reshape(-1,)
                    image_np = image.cpu().detach().numpy()
                    image_np = np.transpose(image_np[:,:3,:,:], (0,2,3,1))
                    image_np = uplot.denormalize_image(image_np, self.train_loader.dataset.img_mean, self.train_loader.dataset.img_std)
                    pixel_pred_np = pixel_pred.cpu().detach().numpy()
                    pixel_gt_np = pixel_gt.cpu().detach().numpy()
                    
                    for j_idx, dataset_idx in enumerate(dataset_indices_np):                    
                        dataset_idx = dataset_indices_np[j_idx]
                        sample_id = self.train_loader.dataset.idx_to_sample_id[dataset_idx]
                        img_plot = image_np[j_idx]

                        pixel_pred_idx = pixel_pred_np[j_idx] * self.train_loader.dataset.size
                        pixel_gt_idx = pixel_gt_np[j_idx] * self.train_loader.dataset.size
                        
                        uplot.plot_predicted_image(self.cnt, img_plot, pixel_pred_idx, pixel_gt_idx, 'train_loss_spike', sample_id)
            
            hist_loss_pixel = loss_pixel_w.item()

            wandb_dict = {'train/train_loss':loss.item(), \
                'train/train_loss_cat': loss_cat_w.item(), \
                'train/train_loss_obj': loss_obj_w.item(), \
                'train/train_loss_scale': loss_scale_w.item(), \
                'train/train_loss_pixel': loss_pixel_w.item(), \
                'train/learning_rate': self.optimizer.param_groups[0]['lr']}

            wandb.log(wandb_dict, step=self.cnt)
            
            # Plot triplet pairs
            if self.cnt % self.args.plot_triplet_every == 0:
                
                image_np = image.cpu().detach().numpy()[:,:3,:,:]
                mask_np = image.cpu().detach().numpy()[:,3,:,:]
                image_np = np.transpose(image_np, (0,2,3,1))
                image_np = uplot.denormalize_image(image_np, self.train_loader.dataset.img_mean, self.train_loader.dataset.img_std)
                pixel_pred_np = pixel_pred.cpu().detach().numpy()
                pixel_gt_np = pixel_gt.cpu().detach().numpy()

                for mask,mask_name in [(mask_cat, "mask_cat"), (mask_id, "mask_id")]:
                    loss_pairs = torch.stack(torch.where(mask), dim=1)
                    plt_pairs_idx = np.random.choice(len(loss_pairs), self.args.triplet_plot_num, replace=False)
                    loss_pairs = loss_pairs[list(plt_pairs_idx)]
                    loss_pairs_idx_in_dataset = dataset_indices[loss_pairs].view((-1,3))
                    
                    j = 0
                    for idx_in_dataset in loss_pairs_idx_in_dataset:
                        j_idxs = loss_pairs[j] # index in batch {0,...,batch_size-1}
                        idx_in_dataset = idx_in_dataset.cpu().numpy().astype(int)  # index in dataset {0,...,len(dataset)-1}

                        fig, axs = plt.subplots(1, 3, figsize=(30,20))  
                        sample_ids = []
                        for i in range(3):
                            sample = self.train_loader.dataset.idx_to_data_dict[idx_in_dataset[i]]
                            sample_ids.append(sample['sample_id'])
                            img = image_np[j_idxs[i]]
                            maski = mask_np[j_idxs[i]]
                            masked_img = uplot.masked_image(img, maski)
                            axs[i].imshow(masked_img)
                            axs[i].set_title('{}_{}'.format(sample['obj_cat'], sample['obj_id']))
                        
                        final_img = uplot.plt_to_image(fig)
                        log_key = '{}/{}_{}_{}'.format(mask_name, epoch, self.cnt, '_'.join(sample_ids))
                        wandb.log({log_key: wandb.Image(final_img)}, step=self.cnt)
                        plt.close()
                        j += 1
            
            # Log info
            if self.cnt % self.args.log_every == 0:
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                    epoch, self.cnt, 100. * batch_idx / len(self.train_loader), loss.item(), \
                        loss_cat_w.item(), loss_obj_w.item(), loss_scale_w.item(), loss_pixel_w.item()))
            
            torch.cuda.empty_cache()

            # Validation iteration
            if self.cnt % self.args.val_every == 0:
                self.model.eval()
                l1,l2,l3,l4 = test.eval_dataset(epoch, self.cnt, self.model, self.device, self.test_loader, self.loss_args, self.test_args, self.args.loss_used)
                self.model.train()
                
                print('Validate Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                    epoch, self.cnt, 100. * batch_idx / len(self.test_loader), l1,l2,l3,l4))

                wandb_dict = {'test/test_loss_cat': l1, \
                    'test/test_loss_obj': l2, \
                    'test/test_loss_scale': l3, \
                    'test/test_loss_pixel': l4}

                wandb.log(wandb_dict, step=self.cnt)
            torch.cuda.empty_cache()
                
            self.cnt += 1
    
        if save_this_epoch(self.args, epoch):
            save_model(self.args.model_save_dir, epoch, self.wandb_run_name, self.model)
        
        if self.scheduler is not None:
            self.scheduler.step()

