from __future__ import print_function

import torch
import numpy as np
import os
import torchvision
import PIL
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import utils.plot_image as uplot 
import utils.transforms as utrans
import utils.utils as uu
import test
import loss.triplet_loss as triploss

def save_this_epoch(args, epoch):

    if args.save_freq < 0:
        return False 
    return epoch % args.save_freq == 0

class Trainer(object):
    def __init__(self, all_args, model, train_loader, test_loader, optimizer, scheduler, device):

        self.args = all_args
        self.wandb_enabled = self.args.wandb.enable
        self.train_args = all_args.training_config 
        self.loss_args = all_args.loss 
        self.test_args = all_args.testing_config
        self.model = model 

        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.device = device
        self.model.train()

        if self.train_args.loss_used == "l1":
            self.criterion_scale = torch.nn.L1Loss()
            self.criterion_pixel = torch.nn.L1Loss()
        else:
            self.criterion_scale = torch.nn.MSELoss()
            self.criterion_pixel = torch.nn.MSELoss()

        self.cnt = -1

        if self.wandb_enabled and not wandb.run is None:
            self.wandb_run_name = wandb.run.name 
        else:
            self.wandb_run_name = uu.get_timestamp()
        
        experiment_save_dir = self.train_args.experiment_save_dir
        uu.create_dir(experiment_save_dir)
        self.this_experiment_dir = os.path.join(experiment_save_dir, self.wandb_run_name)
        uu.create_dir(self.this_experiment_dir)

        self.model_dir = os.path.join(self.this_experiment_dir, "models")
        uu.create_dir(self.model_dir)
        
        self.image_dir = os.path.join(self.this_experiment_dir, "images")
        uu.create_dir(self.image_dir)

        self.prediction_dir = os.path.join(self.this_experiment_dir, "predictions")
        uu.create_dir(self.prediction_dir)

        self.load_model_from()

    def load_model_from(self):
        if self.args.model_config.model_path is not None:
            print("=> Loading model file from: ", self.args.model_config.model_path)
            model_path = os.path.join(self.args.model_config.model_path)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def save_model(self, epoch):
        model_path = os.path.join(self.model_dir, '{}.pth'.format(epoch))
        try:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    }, model_path)
        except:
            print("ERROR: Cannot save model at", model_path)
    
    def train(self, test_only = False):
        start_epoch = self.train_args.start_epoch
        
        if not test_only: 
            self.cnt = 0
            for epoch in range(start_epoch, self.train_args.epochs):
                self.train_epoch(epoch)
        
            if self.train_args.save_at_end:
                self.save_model(self.train_args.epochs)
        
        self.test(self.train_args.epochs, last_epoch=True)

    def train_epoch(self, epoch):

        hist_loss_pixel = 0.0
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            image = data["image"]
            scale_gt = data["scale"]
            pixel_gt = data["center"]
            cat_gt = data["obj_category"]
            id_gt = data["obj_id"]
            dataset_indices = data["idx"]  
            
            # Send model and data to CUDA
            self.model = self.model.to(self.device)
            image = image.to(self.device)
            scale_gt = scale_gt.to(self.device)
            pixel_gt = pixel_gt.to(self.device)
            cat_gt = cat_gt.to(self.device)
            id_gt = id_gt.to(self.device)
            
            img_embed, pose_pred = self.model(image)
            # Position prediction [scale, x, y]
            scale_pred = pose_pred[:,:1]
            pixel_pred = pose_pred[:,1:]
            # Normalize embedding
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

            # DEBUG purpose: During training, there are suddent spikes in loss, plot to see instances
            if epoch <= self.train_args.loss_anormaly_detech_epoch:
                if batch_idx > 0 and loss_pixel_w.item() >= (self.train_args.loss_anormaly_scale - 1) * hist_loss_pixel:
                    print("Spike in training batch: ", batch_idx, loss_pixel_w.item())
                    dataset_indices_np = dataset_indices.cpu().numpy().astype(int).reshape(-1,)
                    image_tensor = image.cpu().detach()[:,:3,:,:]
                    image_tensor = utrans.denormalize(image_tensor, self.train_loader.dataset.img_mean, self.train_loader.dataset.img_std)
                    
                    pixel_pred_np = pixel_pred.cpu().detach().numpy()
                    pixel_gt_np = pixel_gt.cpu().detach().numpy()
                    
                    for idx_in_batch, dataset_idx in enumerate(dataset_indices_np):                    
                        sample_id = self.train_loader.dataset.idx_to_sample_id[dataset_idx]
                        img_plot = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])

                        pixel_pred_idx = pixel_pred_np[idx_in_batch] * self.train_loader.dataset.size
                        pixel_gt_idx = pixel_gt_np[idx_in_batch] * self.train_loader.dataset.size
                        
                        uplot.plot_predicted_image(self.cnt, img_plot, pixel_pred_idx, pixel_gt_idx, enable_wandb = self.wandb_enabled, image_type_name='train_loss_spike', sample_id=sample_id)
            hist_loss_pixel = loss_pixel_w.item()


            if self.wandb_enabled:
                wandb_dict = {'train/train_loss':loss.item(), \
                'train/train_loss_cat': loss_cat_w.item(), \
                'train/train_loss_obj': loss_obj_w.item(), \
                'train/train_loss_scale': loss_scale_w.item(), \
                'train/train_loss_pixel': loss_pixel_w.item(), \
                'train/learning_rate': self.optimizer.param_groups[0]['lr']}

                wandb.log(wandb_dict, step=self.cnt)
            
            # Plot triplet pairs
            if self.cnt % self.train_args.plot_triplet_every == 0:
                image_tensor = image.cpu().detach()[:,:3,:,:]
                image_tensor = utrans.denormalize(image_tensor, self.train_loader.dataset.img_mean, self.train_loader.dataset.img_std)
                mask_tensor = image.cpu().detach()[:,3:,:,:]
        
                cat_gt_np = cat_gt.cpu().detach().numpy()
                id_gt_np = id_gt.cpu().detach().numpy()
                dataset_indices_np = dataset_indices.cpu().numpy().astype(int).reshape(-1,)

                for mask,mask_name in [(mask_cat, "mask_cat"), (mask_id, "mask_id")]:
                    triplets = torch.stack(torch.where(mask), dim=1)
                    plt_pairs_idx = np.random.choice(len(triplets), self.train_args.triplet_plot_num, replace=False)
                    triplets = triplets[list(plt_pairs_idx)]
                    
                    for triplet in triplets:
                        fig, axs = plt.subplots(1, 3, figsize=(30,20))  
                        sample_ids = [self.train_loader.dataset.idx_to_sample_id[int(dataset_indices[idx].item())] for idx in triplet]
                        for i in range(3):
                            idx_in_batch = triplet[i]
                            obj_cat, obj_id = cat_gt_np[idx_in_batch], id_gt_np[idx_in_batch]
                            image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
                            mask_PIL = torchvision.transforms.ToPILImage()(mask_tensor[idx_in_batch])
                            obj_background = PIL.Image.new("RGB", image_PIL.size, 0)
                            masked_image = PIL.Image.composite(image_PIL, obj_background, mask_PIL)
                            axs[i].imshow(np.asarray(masked_image))
                            axs[i].set_title('{}_{}'.format(obj_cat, obj_id))
                        
                        image_name = '{}_{}_{}'.format(epoch, self.cnt, '_'.join(sample_ids))
                        if self.wandb_enabled:
                            final_img = uplot.plt_to_image(fig)
                            log_key = '{}/{}'.format(mask_name, image_name)
                            wandb.log({log_key: wandb.Image(final_img)}, step=self.cnt)
                        else:
                            image_path = os.path.join(self.image_dir, "{}_{}.png".format(mask_name, image_name))
                            plt.savefig(image_path)
                        plt.close()
            
            # Log info
            if self.cnt % self.train_args.log_every == 0:
                print('Train Epoch: {} [{} ({:.0f}%)]\tTotal Loss={:.6f}, Triplet_Loss_Category ({}) = {:.6f}, Triplet_Loss_Object ({}) = {:.6f}, Object_Scale_Loss ({}) = {:.6f}, Object_2D_Center_Loss ({}) = {:.6f}'.format(
                    epoch, self.cnt, 100. * batch_idx / len(self.train_loader), loss.item(), \
                        self.loss_args.lambda_cat, loss_cat_w.item(), \
                        self.loss_args.lambda_obj, loss_obj_w.item(), \
                        self.loss_args.lambda_scale, loss_scale_w.item(), \
                        self.loss_args.lambda_pixel, loss_pixel_w.item()))
            
            torch.cuda.empty_cache()

            # Validation iteration
            if self.cnt % self.train_args.val_every == 0:
                self.test(epoch)
                self.model.train()
            torch.cuda.empty_cache()
                
            self.cnt += 1
    
        if save_this_epoch(self.train_args, epoch):
            self.save_model(epoch)
        
        if self.scheduler is not None:
            self.scheduler.step()
    

    def test(self, epoch, last_epoch=False):
        self.model.eval()
        loss_cat = []
        loss_obj = []

        scale_pred_list = []
        scale_gt_list = []
        pixel_pred_list = []
        pixel_gt_list = []

        embeds = []
        sample_ids_list = []

        test_dataset = self.test_loader.dataset

        plot_step = len(self.test_loader) // self.test_args.num_gt_image_plot
        plot_batch_idx = np.arange(self.test_args.num_gt_image_plot) * plot_step
        
        with torch.no_grad():
            
            for batch_idx, data in enumerate(self.test_loader):
                image = data["image"]
                scale_gt = data["scale"]
                pixel_gt = data["center"]
                cat_gt = data["obj_category"]
                id_gt = data["obj_id"]
                dataset_indices = data["idx"] 
                sample_ids = test_dataset.idx_to_sample_id[dataset_indices.numpy().astype(int)].reshape(-1,)
                sample_ids_list.append(sample_ids)

                model = self.model.to(self.device)
                image = image.to(self.device)

                img_embed, pose_pred = model(image)
                img_embed = img_embed.cpu()

                embeds.append(img_embed)
                image = image.cpu().detach()
                # images.append(image)

                pose_pred = pose_pred.cpu()
                pose_pred = pose_pred.float().detach()
                scale_pred = pose_pred[:,:1]
                pixel_pred = pose_pred[:,1:]

                scale_pred_list.append(scale_pred)
                scale_gt_list.append(scale_gt)
                pixel_pred_list.append(pixel_pred)
                pixel_gt_list.append(pixel_gt)

                # Normalize the image embedding
                img_embed -= img_embed.min(1, keepdim=True)[0]
                img_embed /= img_embed.max(1, keepdim=True)[0]

                _,c_loss = triploss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=self.loss_args.margin, squared=False) #.cpu()
                _,o_loss = triploss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=self.loss_args.margin, squared=False) #.cpu()
                # print("===> Testing: ", batch_idx, c_loss.item()* loss_args.lambda_cat, o_loss.item()* loss_args.lambda_obj)

                # print(torch.nn.MSELoss()(pixel_pred, pixel_gt).item() * loss_args.lambda_pixel)

                if batch_idx in plot_batch_idx and batch_idx != len(self.test_loader)-1:
                    print("===> Plotting Test: ", batch_idx)
                    idx_in_batch = np.random.choice(len(dataset_indices),1)[0]
                    dataset_idx = int(dataset_indices[idx_in_batch].item())
                    pixel_pred_idx = pixel_pred[idx_in_batch].cpu().detach().numpy() * test_dataset.size
                    pixel_gt_idx = pixel_gt[idx_in_batch].cpu().detach().numpy() * test_dataset.size
                    scale_pred_idx = scale_pred[idx_in_batch].item()
                    scale_gt_idx = scale_gt[idx_in_batch].item()
                    
                    image_tensor = image.cpu().detach()[:,:3,:,:]
                    image_tensor = utrans.denormalize(image_tensor, self.train_loader.dataset.img_mean, self.train_loader.dataset.img_std)
                    image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
                    sample_id = test_dataset.idx_to_sample_id[dataset_idx]
                    uplot.plot_predicted_image(self.cnt, image_PIL, pixel_pred_idx, pixel_gt_idx, enable_wandb = self.wandb_enabled, image_type_name='test_pixel_image', image_dir = self.image_dir, sample_id=sample_id, scale_pred_idx = scale_pred_idx, scale_gt_idx = scale_gt_idx)

                loss_cat.append(c_loss.item())
                loss_obj.append(o_loss.item())

                torch.cuda.empty_cache()
        
        all_scale_pred = torch.cat(scale_pred_list, dim=0)
        all_scale_gt = torch.cat(scale_gt_list, dim=0)
        all_pixel_pred = torch.cat(pixel_pred_list, dim=0)
        all_pixel_gt = torch.cat(pixel_gt_list, dim=0)
        # print(all_scale_pred.shape, all_scale_gt.shape, all_pixel_pred.shape, all_pixel_gt.shape)
        
        if last_epoch or epoch % self.test_args.save_prediction_every == 0:
            all_pose = torch.cat([all_scale_pred, all_pixel_pred, all_scale_gt, all_pixel_gt], dim=1)
            pose_path = os.path.join(self.prediction_dir, '{}_pose.npy'.format(epoch))
            np.save(pose_path, all_pose)

            all_embedding = torch.cat(embeds, dim=0).numpy()
            feat_path = os.path.join(self.prediction_dir, '{}_embedding.npy'.format(epoch))
            np.save(feat_path, all_embedding)

            all_sample_ids = np.hstack(sample_ids_list)
            sample_ids_path = os.path.join(self.prediction_dir, '{}_sample_id.npy'.format(epoch))
            np.save(sample_ids_path, all_sample_ids)

        final_loss_cat = np.mean(loss_cat) * self.loss_args.lambda_cat
        final_loss_obj = np.mean(loss_obj) * self.loss_args.lambda_obj

        if self.train_args.loss_used == 'l1':
            final_loss_s = torch.nn.L1Loss()(all_scale_pred, all_scale_gt).item() * self.loss_args.lambda_scale
            final_loss_p = torch.nn.L1Loss()(all_pixel_pred, all_pixel_gt).item() * self.loss_args.lambda_pixel
        else:
            final_loss_s = torch.nn.MSELoss()(all_scale_pred, all_scale_gt).item() * self.loss_args.lambda_scale
            final_loss_p = torch.nn.MSELoss()(all_pixel_pred, all_pixel_gt).item() * self.loss_args.lambda_pixel

    
        print('Validate Epoch: {} , Iteration: {}\tTriplet_Loss_Category ({}) = {:.6f}, Triplet_Loss_Object ({}) = {:.6f}, Object_Scale_Loss ({}) = {:.6f}, Object_2D_Center_Loss ({}) = {:.6f}'.format(
                    epoch, self.cnt, \
                    self.loss_args.lambda_cat, final_loss_cat, \
                    self.loss_args.lambda_obj, final_loss_obj, \
                    self.loss_args.lambda_scale, final_loss_s, \
                    self.loss_args.lambda_pixel, final_loss_p))

        if self.wandb_enabled:
            wandb_dict = {'test/test_loss_cat': final_loss_cat, \
                'test/test_loss_obj': final_loss_obj, \
                'test/test_loss_scale': final_loss_s, \
                'test/test_loss_pixel': final_loss_p}

            wandb.log(wandb_dict, step=self.cnt)



        
