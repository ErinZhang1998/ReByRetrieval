import torch
import numpy as np 
import losses.loss as loss
import wandb
import os 
import utils.transforms as utrans
import torchvision
import utils.plot_image as uplot 
import utils.distributed as du

class TestMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, args):
        self.args = args

        self.loss_cat = 0.0
        self.loss_obj = 0.0
        self.count = 0

        self.acc_dict = dict()

    def reset(self):
        self.loss_cat = 0.0
        self.loss_obj = 0.0
        self.count = 0

        for k,v in self.acc_dict.items():
            self.acc_dict[k] = []

    def update_stats(self, iter_data):
        self.loss_cat += iter_data['loss_cat']
        self.loss_obj += iter_data['loss_obj']
        self.count += 1

        for k,v in iter_data.items():
            l = self.acc_dict.get(v, [])
            l.append(v)
            self.acc_dict[k] = l

    def log_iter_stats(self, iter_data, cnt, image_dir, wandb_enabled=False, plot=False):
        if plot and du.is_master_proc(num_gpus=self.args.num_gpus):
            dataset_indices = iter_data['dataset_indices'].numpy().reshape(-1,)
            pixel_pred = iter_data['pixel_pred']
            pixel_gt = iter_data['pixel_gt']
            scale_pred = iter_data['scale_pred']
            scale_gt = iter_data['scale_gt']
            image = iter_data['image']
            sample_id = iter_data['sample_id']
            
            idx_in_batch = np.random.choice(len(dataset_indices),1)[0]
            dataset_idx = int(dataset_indices[idx_in_batch].item())
            pixel_pred_idx = pixel_pred[idx_in_batch].numpy()
            pixel_pred_idx[0] *= self.args.dataset_config.size_w
            pixel_pred_idx[1] *= self.args.dataset_config.size_h
            
            pixel_gt_idx = pixel_gt[idx_in_batch].numpy()
            pixel_gt_idx[0] *= self.args.dataset_config.size_w
            pixel_gt_idx[1] *= self.args.dataset_config.size_h
            
            scale_pred_idx = scale_pred[idx_in_batch].numpy()
            scale_gt_idx = scale_gt[idx_in_batch].numpy()
            
            image_tensor = image[:,:3,:,:]
            image_tensor = utrans.denormalize(image_tensor, self.args.dataset_config.img_mean, self.args.dataset_config.img_std)
            image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
            
            this_sample_id = [str(int(ele)) for ele in iter_data['sample_id'][idx_in_batch].numpy()]
            this_sample_id = '_'.join(this_sample_id)
            
            uplot.plot_predicted_image(cnt, image_PIL, pixel_pred_idx, pixel_gt_idx, enable_wandb = wandb_enabled, image_type_name='test_pixel_image', image_dir = image_dir, sample_id=this_sample_id, scale_pred_idx = scale_pred_idx, scale_gt_idx = scale_gt_idx)

        self.update_stats(iter_data)

    def finalize_metrics(self, epoch, cnt, prediction_dir):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return 

        all_scale_pred = torch.cat(self.acc_dict['scale_pred'], dim=0)
        all_scale_gt = torch.cat(self.acc_dict['scale_gt'], dim=0)
        all_pixel_pred = torch.cat(self.acc_dict['pixel_pred'], dim=0)
        all_pixel_gt = torch.cat(self.acc_dict['pixel_gt'], dim=0)

        final_loss_cat = (self.loss_cat / self.count) * self.args.loss.lambda_cat
        final_loss_obj = (self.loss_obj / self.count) * self.args.loss.lambda_cat

        
        loss_fun = loss.get_loss_func(self.args.training_config.loss_used)(reduction="mean")
        final_loss_s = loss_fun(all_scale_pred, all_scale_gt).item() * self.args.loss.lambda_scale
        final_loss_p = loss_fun(all_pixel_pred, all_pixel_gt).item() * self.args.loss.lambda_pixel

        print('Validate Epoch: {} , Iteration: {}\tTriplet_Loss_Category ({}) = {:.6f}, Triplet_Loss_Object ({}) = {:.6f}, Object_Scale_Loss ({}) = {:.6f}, Object_2D_Center_Loss ({}) = {:.6f}'.format(
                epoch, cnt, \
                self.args.loss.lambda_cat, final_loss_cat, \
                self.args.loss.lambda_obj, final_loss_obj, \
                self.args.loss.lambda_scale, final_loss_s, \
                self.args.loss.lambda_pixel, final_loss_p))

        if self.args.wandb.enable and not wandb.run is None:
            wandb_dict = {'test/test_loss_cat': final_loss_cat, \
                'test/test_loss_obj': final_loss_obj, \
                'test/test_loss_scale': final_loss_s, \
                'test/test_loss_pixel': final_loss_p}

            wandb.log(wandb_dict, step=cnt)
                
        if epoch == self.args.training_config.epochs or epoch % self.args.testing_config.save_prediction_every == 0 or (not self.args.training_config.train):
            all_pose = torch.cat([all_scale_pred, all_pixel_pred, all_scale_gt, all_pixel_gt], dim=1)
            pose_path = os.path.join(prediction_dir, '{}_pose.npy'.format(epoch))
            np.save(pose_path, all_pose)

            all_embedding = torch.cat(self.acc_dict['embeds'], dim=0).numpy()
            feat_path = os.path.join(prediction_dir, '{}_embedding.npy'.format(epoch))
            np.save(feat_path, all_embedding)

            all_sample_ids = np.hstack(self.acc_dict['sample_id'])
            sample_ids_path = os.path.join(prediction_dir, '{}_sample_id.npy'.format(epoch))
            np.save(sample_ids_path, all_sample_ids)

            all_area_types = np.hstack(self.acc_dict['area_type'])
            sample_ids_path = os.path.join(prediction_dir, '{}_area_types.npy'.format(epoch))
            np.save(sample_ids_path, all_area_types)

        