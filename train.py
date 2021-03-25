from __future__ import print_function

import torch
import numpy as np

import utils.utils as uu
import test
import loss.triplet_loss as triploss

import os
import datetime
import time

import torchvision.utils as vutils
from tensorboardX import SummaryWriter


def save_this_epoch(args, epoch):

    if args.save_freq < 0:
        return False 
    return epoch % args.save_freq == 0

def get_timestamp():
    ts = time.time()
    timenow = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    return timenow

def save_model(model_save_dir, epoch, model_name, model, train_timestamp):
    if not os.path.join(model_save_dir, 'models'):
        os.mkdir(os.path.join(model_save_dir, 'models'))
    
    model_dir = os.path.join(model_save_dir, 'models', '{}_{}'.format(model_name, train_timestamp))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 

    model_path = os.path.join(model_dir, '{}_{}.pth'.format(model_name, epoch))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, model_path)

class Trainer(object):
    def __init__(self, all_args, model, model_name, train_loader, test_loader, optimizer, scheduler, device):

        self.args = all_args.training_config 
        self.loss_args = all_args.loss 
        self.test_args = all_args.testing_config
        self.model = model 
        self.model_name = model_name 
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.device = device

        self.timenow = get_timestamp()
        writer_summary_folder = os.path.join(self.args.tensorboard_save_dir, 'runs/{}_{}'.format(model_name, self.timenow))
        self.writer = SummaryWriter(writer_summary_folder)

        self.model.train()

        self.criterion_scale = torch.nn.MSELoss()
        self.criterion_pixel = torch.nn.MSELoss()

        self.cnt = 0 
    
    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
        
        if self.args.save_at_end:
            save_model(self.args.model_save_dir, self.args.epochs, self.model_name, self.model, self.timenow)

        
        l1,l2,l3,l4 = test.eval_dataset(self.cnt, self.loss_args, self.model, self.device, self.test_loader, self.writer, self.test_args)
        self.writer.close()
        return l1,l2,l3,l4,self.cnt


    def train_epoch(self, epoch):
        for batch_idx, (image, scale_info, pixel_info, cat_info, id_info) in enumerate(self.train_loader):
            self.model = self.model.to(self.device)

            image = image.to(self.device)
            scale_info = scale_info.to(self.device)
            pixel_info = pixel_info.to(self.device)
            cat_info = cat_info.to(self.device)
            id_info = id_info.to(self.device)
            
            self.optimizer.zero_grad()
            img_embed, pose_pred = self.model(image)
            scale_pred = pose_pred[:,:1]
            pixel_pred = pose_pred[:,1:]
            img_embed -= img_embed.min(1, keepdim=True)[0]
            img_embed /= img_embed.max(1, keepdim=True)[0]


            loss_cat = triploss.batch_all_triplet_loss(labels=cat_info, embeddings=img_embed, margin=self.loss_args.margin, squared=False)
            loss_obj = triploss.batch_all_triplet_loss(labels=id_info, embeddings=img_embed, margin=self.loss_args.margin, squared=False)
            loss_scale = self.criterion_scale(scale_pred, scale_info)
            loss_pixel = self.criterion_pixel(pixel_pred, pixel_info)

            loss = self.loss_args.lambda_cat * loss_cat + \
                self.loss_args.lambda_obj * loss_obj + \
                self.loss_args.lambda_scale * loss_scale + \
                self.loss_args.lambda_pixel * loss_pixel
            loss.backward()
            
            self.optimizer.step()
        
        
            self.writer.add_scalar('data/train_loss', loss.item(), self.cnt)
            self.writer.add_scalar('data/train_loss_cat', loss_cat.item(), self.cnt)
            self.writer.add_scalar('data/train_losss_obj', loss_obj.item(), self.cnt)
            self.writer.add_scalar('data/train_loss_scale', loss_scale.item(), self.cnt)
            self.writer.add_scalar('data/train_loss_pixel', loss_pixel.item(), self.cnt)
            self.writer.add_scalar('data/learning_rate', self.optimizer.param_groups[0]['lr'], self.cnt)
            
            # Log info
            if self.cnt % self.args.log_every == 0:
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                    epoch, self.cnt, 100. * batch_idx / len(self.train_loader), loss.item(), \
                        loss_cat.item(), loss_obj.item(), loss_scale.item(), loss_pixel.item()))
            
            torch.cuda.empty_cache()

            # Validation iteration
            if self.cnt % self.args.val_every == 0:
                self.model.eval()
                l1,l2,l3,l4 = test.eval_dataset(self.cnt, self.loss_args, self.model, self.device, self.test_loader, self.writer, self.test_args)
                self.model.train()
                
                print('Validate Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                    epoch, self.cnt, 100. * batch_idx / len(self.test_loader), l1,l2,l3,l4))

                self.writer.add_scalar('data/test_loss_cat', l1, self.cnt)
                self.writer.add_scalar('data/test_loss_obj', l2, self.cnt)
                self.writer.add_scalar('data/test_loss_scale', l3, self.cnt)
                self.writer.add_scalar('data/test_loss_pixel', l4, self.cnt)
            torch.cuda.empty_cache()
                
            self.cnt += 1
    
        if save_this_epoch(self.args, epoch):
            save_model(self.args.model_save_dir, epoch, self.model_name, self.model, self.timenow)
        
        if self.scheduler is not None:
            self.scheduler.step()

