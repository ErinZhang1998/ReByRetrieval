import torch
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image
import copy
import wandb
import os 

import utils.utils as uu
import loss.triplet_loss as triploss
import utils.plot_image as uplot

def eval_dataset(epoch, cnt, model, device, test_loader, loss_args, test_args, loss_used, last_epoch=False):
    # np.random.seed(129)
    loss_cat = []
    loss_obj = []

    scale_pred_list = []
    scale_gt_list = []
    pixel_pred_list = []
    pixel_gt_list = []

    embeds = []
    indices = []
    sample_ids = []
    # images = []

    test_dataset = test_loader.dataset
    
    with torch.no_grad():
        
        for batch_idx, data in enumerate(test_loader):
            image = data[0]
            scale_gt = data[1]
            pixel_gt = data[2]
            cat_gt = data[3]
            id_gt = data[4]
            dataset_indices = data[5]

            model = model.to(device)
            image = image.to(device)

            img_embed, pose_pred = model(image)
            img_embed = img_embed.cpu()

            embeds.append(img_embed)
            indices.append(dataset_indices)
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

            _,c_loss = triploss.batch_all_triplet_loss(labels=cat_gt, embeddings=img_embed, margin=loss_args.margin, squared=False) #.cpu()
            _,o_loss = triploss.batch_all_triplet_loss(labels=id_gt, embeddings=img_embed, margin=loss_args.margin, squared=False) #.cpu()

            if batch_idx % test_args.plot_gt_image_every == 0 and batch_idx != len(test_loader)-1:
                print("===> Plotting Test: ", batch_idx)
                j_idx = np.random.choice(len(dataset_indices),1)[0]
                dataset_idx = dataset_indices.reshape(-1,)[j_idx].item()
                sample = test_dataset.idx_to_data_dict[dataset_idx]
                sample_id = sample['sample_id']
                img_plot = mpimg.imread(sample['rgb_all_path'])
                
                pixel_pred_idx = pixel_pred.numpy()[j_idx].reshape(-1,)
                pixel_pred_idx[0] *= test_dataset.img_w
                pixel_pred_idx[1] *= test_dataset.img_h
                pixel_gt_idx = copy.deepcopy(sample['object_center'].reshape(-1,))
                pixel_gt_idx[0] = test_dataset.img_w - pixel_gt_idx[0]
                scale_pred_idx = scale_pred[j_idx].item()
                scale_gt_idx = scale_gt[j_idx].item()
                
                uplot.plot_predicted_image(cnt, img_plot, pixel_pred_idx, pixel_gt_idx, 'test_pixel_image', sample_id, scale_pred_idx, scale_gt_idx)

            loss_cat.append(c_loss.item())
            loss_obj.append(o_loss.item())

            torch.cuda.empty_cache()
    
    scale_pred = torch.cat(scale_pred_list, dim=0)
    scale_gt = torch.cat(scale_gt_list, dim=0)
    pixel_pred = torch.cat(pixel_pred_list, dim=0)
    pixel_gt = torch.cat(pixel_gt_list, dim=0)
    
    if last_epoch or epoch % test_args.save_prediction_every == 0:
        all_pose = torch.cat([scale_pred, pixel_pred, scale_gt, pixel_gt], dim=1)
        pose_path = os.path.join(test_args.save_prediction_dir, '{}_{}_pose.npy'.format(wandb.run.name, epoch))
        np.save(pose_path, all_pose)

        all_embedding = torch.cat(embeds, dim=0).numpy()
        all_indices = torch.cat(indices, dim=0).numpy()
        # all_images = torch.cat(images, dim=0).numpy()
        feat_path = os.path.join(test_args.save_prediction_dir, '{}_{}.npy'.format(wandb.run.name, epoch))
        np.save(feat_path, all_embedding)
        ind_path = os.path.join(test_args.save_prediction_dir, '{}_{}_index.npy'.format(wandb.run.name, epoch))
        np.save(ind_path, all_indices)
        # image_path = os.path.join(test_args.save_prediction_dir, '{}_{}_image.npy'.format(wandb.run.name, epoch))
        # np.save(image_path, all_images)

    final_loss_cat = np.mean(loss_cat) * loss_args.lambda_cat
    final_loss_obj = np.mean(loss_obj) * loss_args.lambda_obj

    if loss_used == 'l1':
        final_loss_s = torch.nn.L1Loss()(scale_pred, scale_gt).item() * loss_args.lambda_scale
        final_loss_p = torch.nn.L1Loss()(pixel_pred, pixel_gt).item() * loss_args.lambda_pixel
    else:
        final_loss_s = torch.nn.MSELoss()(scale_pred, scale_gt).item() * loss_args.lambda_scale
        final_loss_p = torch.nn.MSELoss()(pixel_pred, pixel_gt).item() * loss_args.lambda_pixel

    
    return final_loss_cat, final_loss_obj, final_loss_s, final_loss_p

