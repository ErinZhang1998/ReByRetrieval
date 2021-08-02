from logging import Logger
import torch
import numpy as np 
import losses.loss as loss
import wandb
import os 
import utils.transforms as utrans
import torchvision
import utils.plot_image as uplot 
import utils.distributed as du
import utils.logging as logging 
import losses.loss as loss
import utils.metric as metric 

logger = logging.get_logger(__name__)

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

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

    def update_stats(self, iter_data, cnt):
        
        self.loss_cat += iter_data['loss_cat']
        self.loss_obj += iter_data['loss_obj']
        self.count += 1
        # if du.is_master_proc(num_gpus=self.args.num_gpus):
        #     final_loss_cat = (self.loss_cat / self.count) * self.args.loss.lambda_cat
        #     final_loss_obj = (self.loss_obj / self.count) * self.args.loss.lambda_cat
        #     logger.info('Validate Iteration: {}\tTriplet_Loss_Category ({}) = {:.6f}, Triplet_Loss_Object ({}) = {:.6f}'.format(
        #         cnt, \
        #         self.args.loss.lambda_cat, final_loss_cat, \
        #         self.args.loss.lambda_obj, final_loss_obj))
        for k,v in iter_data.items():
            l = self.acc_dict.get(k, [])
            l.append(v)
            self.acc_dict[k] = l

    def log_iter_stats(self, iter_data, cnt, batch_idx, image_dir, wandb_enabled=False, plot=False):
        if plot and du.is_master_proc(num_gpus=self.args.num_gpus):
            
            scale_pred = iter_data['scale_pred']
            scale_gt = iter_data['scale_gt']
            image = iter_data['image']
            sample_id = iter_data['sample_id']
            
            idx_in_batch = np.random.choice(len(sample_id),1)[0]

            if self.args.model_config.predict_center: 
                pixel_pred = iter_data['pixel_pred']
                pixel_gt = iter_data['pixel_gt']
                pixel_pred_idx = pixel_pred[idx_in_batch].numpy()
                pixel_pred_idx[0] *= self.args.dataset_config.size_w
                pixel_pred_idx[1] *= self.args.dataset_config.size_h
            
                pixel_gt_idx = pixel_gt[idx_in_batch].numpy()
                pixel_gt_idx[0] *= self.args.dataset_config.size_w
                pixel_gt_idx[1] *= self.args.dataset_config.size_h
            else:
                pixel_pred_idx, pixel_gt_idx = None, None
            
            scale_pred_idx = scale_pred[idx_in_batch].numpy()
            scale_gt_idx = scale_gt[idx_in_batch].numpy()
            
            image_tensor = image[:,:3,:,:]
            image_tensor = utrans.denormalize(image_tensor, self.args.dataset_config.img_mean, self.args.dataset_config.img_std)
            image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
            
            this_sample_id = [str(int(ele)) for ele in sample_id[idx_in_batch].numpy()]
            this_sample_id = '_'.join(this_sample_id)
            
            uplot.plot_predicted_image(cnt, image_PIL, pixel_pred_idx, pixel_gt_idx, enable_wandb = wandb_enabled, image_type_name='test_pixel_image', image_dir = image_dir, sample_id=this_sample_id, scale_pred_idx = scale_pred_idx, scale_gt_idx = scale_gt_idx)

        self.update_stats(iter_data, cnt)
    

    def calculate_acc(self, features, cat_label, id_label):
        features = torch.FloatTensor(features)
        features = features.cuda()
        pairwise_dist = loss.pariwise_distances(features, squared=False).cpu()
        arg_sorted_dist = np.argsort(pairwise_dist.numpy(), axis=1)
        # print(features.shape, cat_label.shape, id_label.shape, arg_sorted_dist.shape)
        # import pdb; pdb.set_trace()
        cat_mAP_1 = metric.mapk(cat_label.reshape(-1,1), cat_label[arg_sorted_dist[:,1:]], k=1)
        cat_mAP_5 = metric.mapk(cat_label.reshape(-1,1), cat_label[arg_sorted_dist[:,1:]], k=5)
        cat_mAP_10 = metric.mapk(cat_label.reshape(-1,1), cat_label[arg_sorted_dist[:,1:]], k=10)
        
        object_mAP_1 = metric.mapk(id_label.reshape(-1,1), id_label[arg_sorted_dist[:,1:]], k=1)
        object_mAP_5 = metric.mapk(id_label.reshape(-1,1), id_label[arg_sorted_dist[:,1:]], k=5)
        object_mAP_10 = metric.mapk(id_label.reshape(-1,1), id_label[arg_sorted_dist[:,1:]], k=10) 
        
        # cat_acc_1 = metric.acc_topk_with_dist(cat_label, arg_sorted_dist, 1)
        # cat_acc_5 = metric.acc_topk_with_dist(cat_label, arg_sorted_dist, 5)
        # id_acc_1 = metric.acc_topk_with_dist(id_label, arg_sorted_dist, 1)
        # id_acc_5 = metric.acc_topk_with_dist(id_label, arg_sorted_dist, 5)
        
        # acc_dict = {
        #     'test/category Acc@1' : cat_acc_1,
        #     'test/category Acc@5' : cat_acc_5,
        #     'test/object Acc@1' : id_acc_1,
        #     'test/object Acc@5' : id_acc_5, 
        # }
        mapk_dict = {
            'test/category mAP@1' : cat_mAP_1,
            'test/category mAP@5' : cat_mAP_5,
            'test/category mAP@10' : cat_mAP_10,
            'test/object mAP@1' : object_mAP_1,
            'test/object mAP@5' : object_mAP_5,
            'test/object mAP@10' : object_mAP_10,
        }
        return mapk_dict


    def finalize_metrics(self, epoch, cnt, prediction_dir):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return 

        all_scale_pred = torch.cat(self.acc_dict['scale_pred'], dim=0)
        all_scale_gt = torch.cat(self.acc_dict['scale_gt'], dim=0)

        final_loss_cat = (self.loss_cat / self.count) * self.args.loss.lambda_cat
        final_loss_obj = (self.loss_obj / self.count) * self.args.loss.lambda_cat
        loss_fun = loss.get_loss_func(self.args.training_config.loss_used)(reduction="mean")
        final_loss_s = loss_fun(all_scale_pred, all_scale_gt).item() * self.args.loss.lambda_scale
        
        if self.args.model_config.predict_center: 
            all_pixel_pred = torch.cat(self.acc_dict['pixel_pred'], dim=0)
            all_pixel_gt = torch.cat(self.acc_dict['pixel_gt'], dim=0)
            final_loss_p = loss_fun(all_pixel_pred, all_pixel_gt).item() * self.args.loss.lambda_pixel
        else:
            final_loss_p = -1
       
        logger.info('\n')
        logger.info('Validate Epoch: {} , Iteration: {}'.format(epoch, cnt))
        logger.info('\tTriplet_Loss_Category ({}) = {:.6f}'.format(self.args.loss.lambda_cat, final_loss_cat))
        logger.info('\tTriplet_Loss_Object ({}) = {:.6f}'.format(self.args.loss.lambda_obj, final_loss_obj))
        logger.info('\tObject_Scale_Loss ({}) = {:.6f}'.format(self.args.loss.lambda_scale, final_loss_s))
        if self.args.model_config.predict_center: 
            logger.info('\tObject_2D_Center_Loss ({}) = {:.6f}'.format(self.args.loss.lambda_pixel, final_loss_p))
        
        all_embedding = torch.cat(self.acc_dict['embeds'], dim=0).numpy()
        all_cat_label = torch.cat(self.acc_dict['cat_gt'], dim=0).numpy()
        all_id_label = torch.cat(self.acc_dict['id_gt'], dim=0).numpy()
        acc_dict = self.calculate_acc(all_embedding, all_cat_label, all_id_label)

        if self.args.wandb.enable and not wandb.run is None:
            wandb_dict = {
                'test/test_loss_cat': final_loss_cat,
                'test/test_loss_obj': final_loss_obj,
                'test/test_loss_scale': final_loss_s,
            }
            if self.args.model_config.predict_center: 
                wandb_dict.update({'test/test_loss_pixel': final_loss_p})
            wandb_dict.update(acc_dict)
            wandb.log(wandb_dict, step=cnt)
        else:
            for k,v in acc_dict.items():
                logger.info('\t{} = {:.6f}'.format(k, v))
        
        if epoch == self.args.training_config.epochs or epoch % self.args.testing_config.save_prediction_every == 0 or (not self.args.training_config.train):
            if self.args.model_config.predict_center: 
                all_pose = torch.cat([all_scale_pred, all_pixel_pred, all_scale_gt, all_pixel_gt], dim=1)
            else:
                all_pose = torch.cat([all_scale_pred, all_scale_gt], dim=1)
            pose_path = os.path.join(prediction_dir, '{}_pose.npy'.format(epoch))
            np.save(pose_path, all_pose)
            
            feat_path = os.path.join(prediction_dir, '{}_embedding.npy'.format(epoch))
            np.save(feat_path, all_embedding)

            all_sample_ids = np.vstack(self.acc_dict['sample_id'])
            sample_ids_path = os.path.join(prediction_dir, '{}_sample_id.npy'.format(epoch))
            np.save(sample_ids_path, all_sample_ids)

            all_area_types = np.vstack(self.acc_dict['area_type'])
            sample_ids_path = os.path.join(prediction_dir, '{}_area_types.npy'.format(epoch))
            np.save(sample_ids_path, all_area_types)


class FeatureExtractMeter(object):

    def __init__(self, args):
        self.args = args
        self.acc_dict = dict()

    def reset(self):
        for k,v in self.acc_dict.items():
            self.acc_dict[k] = []

    def update_stats(self, iter_data, cnt):
        for k,v in iter_data.items():
            l = self.acc_dict.get(k, [])
            l.append(v)
            self.acc_dict[k] = l

    def log_iter_stats(self, iter_data, cnt, batch_idx, image_dir=None, wandb_enabled=False, plot=False):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return

        if batch_idx % self.args.testing_config.log_every == 0:
            logger.info("Iter: {}, Log batch {} stats in FeatureExtractMeter".format(cnt, batch_idx))
        self.update_stats(iter_data, cnt)

    def finalize_metrics(self, epoch, cnt, prediction_dir):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return 

        all_scale_pred = torch.cat(self.acc_dict['scale_pred'], dim=0)
        all_scale_gt = torch.cat(self.acc_dict['scale_gt'], dim=0)
        all_embedding = torch.cat(self.acc_dict['embeds'], dim=0).numpy()
        # all_cat_label = torch.cat(self.acc_dict['cat_gt'], dim=0).numpy()
        # all_id_label = torch.cat(self.acc_dict['id_gt'], dim=0).numpy()

        if self.args.model_config.predict_center: 
            all_pixel_pred = torch.cat(self.acc_dict['pixel_pred'], dim=0)
            all_pixel_gt = torch.cat(self.acc_dict['pixel_gt'], dim=0)
            all_pose = torch.cat([all_scale_pred, all_pixel_pred, all_scale_gt, all_pixel_gt], dim=1)
        else:
            all_pose = torch.cat([all_scale_pred, all_scale_gt], dim=1)

        pose_path = os.path.join(prediction_dir, '{}_pose.npy'.format(epoch))
        np.save(pose_path, all_pose)
        
        feat_path = os.path.join(prediction_dir, '{}_embedding.npy'.format(epoch))
        np.save(feat_path, all_embedding)

        all_sample_ids = np.vstack(self.acc_dict['sample_id'])
        sample_ids_path = os.path.join(prediction_dir, '{}_sample_id.npy'.format(epoch))
        np.save(sample_ids_path, all_sample_ids)

        all_area_types = np.vstack(self.acc_dict['area_type'])
        sample_ids_path = os.path.join(prediction_dir, '{}_area_types.npy'.format(epoch))
        np.save(sample_ids_path, all_area_types)

        