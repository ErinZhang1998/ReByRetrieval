from __future__ import print_function

import torch
import numpy as np

import utils.utils as this_utils

import os
import datetime
import time

import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from incat_dataset import *
import resnet_pretrain 

train_dir_root = "/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set"
test_dir_root = "/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/testing_set"


def save_this_epoch(args, epoch):

    if args.save_freq < 0:
        return False 
    return epoch % args.save_freq == 0

def get_timestamp():
    ts = time.time()
    timenow = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    return timenow

def save_model(epoch, model_name, model, train_timestamp):

    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, 'models', '{}_{}'.format(model_name, train_timestamp))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 

    model_path = os.path.join(model_dir, '{}_{}.pth'.format(model_name, epoch))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, model_path)
    
def train(args, model, optimizer, scheduler=None, model_name='resnet50', dataset_size = 227):
    
    timenow = get_timestamp()
    writer_summary_folder = os.path.join(os.getcwd(), 'runs/{}_{}'.format(model_name, timenow))
    writer = SummaryWriter(writer_summary_folder)
    
    train_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/training_set', train=True, batch_size=args.batch_size, \
                                         split='train', \
                                        dataset_size = dataset_size)
    test_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/testing_set', train=False, batch_size=args.test_batch_size, \
                                        split='test', \
                                       dataset_size = dataset_size)
 
    model.train()
    # model = model.to(args.device)
    
    selector = this_utils.BatchHardTripletSelector()
    criterion_embed = torch.nn.TripletMarginLoss(margin = 1, p = 2)
    criterion_scale = torch.nn.MSELoss()
    criterion_orient = torch.nn.MSELoss()
    criterion_pixel = torch.nn.MSELoss()
    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (image, scale_info, orient_info, pixel_info, cat_info) in enumerate(train_loader):
            model = model.to(args.device)
            image, scale_info, orient_info, pixel_info, cat_info = image.to(args.device), scale_info.to(args.device), orient_info.to(args.device), pixel_info.to(args.device), cat_info.to(args.device)
            
            optimizer.zero_grad()
            img_embed, pose_pred = model(image)
            anchor, positives, negatives = selector(img_embed, cat_info.view(-1,))

            # print(scale_info, orient_info, pixel_info)
            # print(pose_pred)

            loss_embed = criterion_embed(anchor, positives, negatives)
            loss_scale = criterion_scale(pose_pred[:,:1], scale_info)
            loss_orient = criterion_orient(pose_pred[:,1:5], orient_info)
            loss_pixel = criterion_pixel(pose_pred[:,5:], pixel_info)

            loss = loss_embed + loss_scale + loss_orient + loss_pixel
            loss.backward()
            optimizer.step()
        
        
            writer.add_scalar('data/train_loss', loss.item(), cnt)
            writer.add_scalar('data/train_loss_scale', loss_scale.item(), cnt)
            writer.add_scalar('data/train_loss_orient', loss_orient.item(), cnt)
            writer.add_scalar('data/train_loss_pixel', loss_pixel.item(), cnt)
            writer.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'], cnt)
            
            # Log info
            if cnt % args.log_every == 0:
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item(), \
                        loss_embed.item(), loss_scale.item(), loss_orient.item(), loss_pixel.item()))
            
            torch.cuda.empty_cache()
            del image
            del scale_info 
            del orient_info 
            del pixel_info 
            del cat_info

            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                scale_loss, orient_loss, pixel_loss = this_utils.eval_dataset(model, args.device, test_loader)
                model.train()
                
                print('Validate Epoch: {} [{} ({:.0f}%)]\tscale_loss:{:.6f}\torient_loss:{:.6f}\tpixel_loss:{:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(test_loader), scale_loss, orient_loss, pixel_loss))
                writer.add_scalar('data/test_loss_scale', scale_loss, cnt)
                writer.add_scalar('data/test_loss_orient', orient_loss, cnt)
                writer.add_scalar('data/test_loss_pixel', pixel_loss, cnt)

                
            cnt += 1
             
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model, timenow)
        if scheduler is not None:
            scheduler.step()
    
    if args.save_at_end:
        save_model(args.epochs, model_name, model, timenow)

    # Validation iteration
    test_loader = this_utils.get_data_loader('/media/xiaoyuz1/hdd5/xiaoyuz1/data/cluttered_datasets/testing_set', train=False, batch_size=args.test_batch_size, \
                                        split='test', \
                                       dataset_size = dataset_size)
    scale_loss, orient_loss, pixel_loss = this_utils.eval_dataset(model, args.device, test_loader)
    
    #writer.export_scalars_to_json("./all_scalars_{}.json".format(model_name))
    writer.close()
    return scale_loss, orient_loss, pixel_loss
