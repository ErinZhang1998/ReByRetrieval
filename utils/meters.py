import torch
import numpy as np 
import wandb
import os 
import utils.transforms as utrans
import torchvision
import utils.plot_image as uplot 
import utils.distributed as du
import torch.distributed as dist


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
        self.contrastive_loss_dict = dict()
        self.acc_dict = dict()

    def reset(self):
        for k in self.contrastive_loss_dict.keys():
            self.contrastive_loss_dict[k] = (0.0, 0.0)

        for k in self.acc_dict.keys():
            self.acc_dict[k] = []

    def update_acc_dict(self, iter_data):
        for k,v in iter_data.items():
            print(dist.get_rank() % self.args.num_gpus, " update_acc_dict, ", k)
            if k.startswith('contrastive_'):
                loss_acc, count = self.contrastive_loss_dict.get(k, (0.0, 0.0))
                loss_acc += v[0]
                count += v[1]
                self.contrastive_loss_dict[k] = (loss_acc, count)
            else:
                l = self.acc_dict.get(k, [])
                l.append(v)
                self.acc_dict[k] = l

    def plot_prediction(
            self, 
            iter_data, 
            cnt, 
            image_dir, 
            wandb_enabled=False, 
        ):
        required_keys = ['image', 'sample_id', 'scale_pred', 'scale']
        for key in required_keys:
            if key not in iter_data:
                return
        image = iter_data['image']
        sample_id = iter_data['sample_id']
        idx_in_batch = np.random.choice(len(sample_id),1)[0]

        image_tensor = image[:,:3,:,:]
        image_tensor = utrans.denormalize(image_tensor, self.args.dataset_config.img_mean, self.args.dataset_config.img_std)
        image_PIL = torchvision.transforms.ToPILImage()(image_tensor[idx_in_batch])
        
        sample_id_i = [str(int(ele)) for ele in sample_id[idx_in_batch].numpy()]
        sample_id_i = '_'.join(sample_id_i)
        
        if 'center_pred' in iter_data and 'center' in iter_data:                
            pixel_pred_idx = iter_data['center_pred'][idx_in_batch]
            pixel_pred_idx[0] *= self.args.dataset_config.size_w
            pixel_pred_idx[1] *= self.args.dataset_config.size_h
        
            pixel_gt_idx = iter_data['center'][idx_in_batch]
            pixel_gt_idx[0] *= self.args.dataset_config.size_w
            pixel_gt_idx[1] *= self.args.dataset_config.size_h
        else:
            pixel_pred_idx, pixel_gt_idx = None, None
        
        uplot.plot_predicted_image(
            cnt, 
            image_PIL, 
            pixel_pred_idx, 
            pixel_gt_idx, 
            enable_wandb = wandb_enabled, 
            image_type_name='test_pixel_image', 
            image_dir = image_dir, 
            sample_id=sample_id_i, 
            scale_pred_idx = iter_data['scale_pred'][idx_in_batch].numpy(), 
            scale_gt_idx = iter_data['scale'][idx_in_batch].numpy(),
        )
    
    def log_iter_stats(
            self, 
            iter_data, 
            cnt, 
            batch_idx, 
            image_dir, 
            wandb_enabled=False, 
            plot=False,
        ):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return
        if plot and du.is_master_proc(num_gpus=self.args.num_gpus):
            self.plot_prediction(
                iter_data, 
                cnt, 
                image_dir, 
                wandb_enabled=wandb_enabled, 
            )
        print(dist.get_rank() % self.args.num_gpus, " before update_acc_dict ")
        self.update_acc_dict(iter_data)
    
    def calculate_mapk(self, loss_key, features, label):
        features = torch.FloatTensor(features)
        features = features.cuda()
        pairwise_dist = loss.pariwise_distances(features, squared=False).cpu()
        arg_sorted_dist = np.argsort(pairwise_dist.numpy(), axis=1)
        
        gt_label = label.reshape(-1,1)
        pred_label = label[arg_sorted_dist[:,1:]]

        mAP_1 = metric.mapk(gt_label, pred_label, k=1)
        mAP_5 = metric.mapk(gt_label, pred_label, k=5)
        mAP_10 = metric.mapk(gt_label, pred_label, k=10)
        
        mapk_dict = {
            f'test/{loss_key} mAP@1' : mAP_1,
            f'test/{loss_key} mAP@5' : mAP_5,
            f'test/{loss_key} mAP@10' : mAP_10,
        }
        return mapk_dict

    def save_prediction(self, epoch, cnt, prediction_dir):
        all_save_keys = list(set(self.args.model_config.model_return + self.args.training_config.gt))
        for key in all_save_keys:
            value = torch.cat(self.acc_dict[key], dim=0) #(batch_size, x)
            value = value.numpy()
            fname = os.path.join(prediction_dir, f'{epoch}_{key}.npy')
            np.save(fname, value)
    
    def finalize_metrics(self, epoch, cnt, prediction_dir):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return 
        logger.info('\n')
        logger.info('Validate Epoch: {} , Iteration: {}'.format(epoch, cnt))

        training_config = self.args.training_config
        wandb_dict = {}
        for loss_idx, loss_fn_name in enumerate(training_config.loss_fn):
            gt_key = training_config.gt[loss_idx]
            pred_key = self.args.model_config.model_return[loss_idx]
            loss_weight = training_config.weight[loss_idx]

            gt_val = torch.cat(self.acc_dict[gt_key], dim=0)
            pred_val = torch.cat(self.acc_dict[pred_key], dim=0)

            if loss_fn_name == 'triplet_loss':
                contrastive_key = f'contrastive_{gt_key}'
                loss_value, count = self.contrastive_loss_dict[contrastive_key]
                loss_value = loss_value / count
            else:
                loss_func = loss.get_loss_func(loss_fn_name)
                loss_value = loss_func(pred_val, gt_val).item()
            
            loss_key = f'{loss_fn_name}_{gt_key}'
            logger.info(
                '\tLoss_fn_name={}, Loss_gt_name={}, Loss_pred_key={}, Loss_weight={}, Loss={:.6f}'.format(
                    loss_fn_name, 
                    gt_key,
                    pred_key,
                    loss_weight,
                    loss_value * loss_weight,
                )
            )
            wandb_dict[f'test/{loss_key}'] = loss_value * loss_weight
            if loss_fn_name == 'triplet_loss':
                mapk_dict = self.calculate_mapk(loss_key, pred_val.numpy(), gt_val.numpy())
                wandb_dict.update(mapk_dict)
                for k,v in mapk_dict.items():
                    logger.info('\t{} = {:.6f}'.format(k, v))

        if self.args.wandb.enable and not wandb.run is None:
            wandb.log(wandb_dict, step=cnt)
        # if epoch == self.args.training_config.epochs or epoch % self.args.testing_config.save_prediction_every == 0 or (not self.args.training_config.train):
        self.save_prediction(epoch, cnt, prediction_dir)
        
        # all_embedding = torch.cat(self.acc_dict['embeds'], dim=0).numpy()
        # all_cat_label = torch.cat(self.acc_dict['cat_gt'], dim=0).numpy()
        # all_id_label = torch.cat(self.acc_dict['id_gt'], dim=0).numpy()
        # acc_dict = self.calculate_acc(all_embedding, all_cat_label, all_id_label)

        
        # if epoch == self.args.training_config.epochs or epoch % self.args.testing_config.save_prediction_every == 0 or (not self.args.training_config.train):
        #     if self.args.model_config.predict_center: 
        #         all_pose = torch.cat([all_scale_pred, all_pixel_pred, all_scale_gt, all_pixel_gt], dim=1)
        #     else:
        #         all_pose = torch.cat([all_scale_pred, all_scale_gt], dim=1)
        #     pose_path = os.path.join(prediction_dir, '{}_pose.npy'.format(epoch))
        #     np.save(pose_path, all_pose)
            
        #     feat_path = os.path.join(prediction_dir, '{}_embedding.npy'.format(epoch))
        #     np.save(feat_path, all_embedding)

        #     all_sample_ids = np.vstack(self.acc_dict['sample_id'])
        #     sample_ids_path = os.path.join(prediction_dir, '{}_sample_id.npy'.format(epoch))
        #     np.save(sample_ids_path, all_sample_ids)

        #     all_area_types = np.vstack(self.acc_dict['area_type'])
        #     sample_ids_path = os.path.join(prediction_dir, '{}_area_types.npy'.format(epoch))
        #     np.save(sample_ids_path, all_area_types)


class FeatureExtractMeter(object):

    def __init__(self, args):
        self.args = args
        self.acc_dict = dict()

    def reset(self):
        for k,v in self.acc_dict.items():
            self.acc_dict[k] = []

    def update_acc_dict(self, iter_data, cnt):
        for k,v in iter_data.items():
            l = self.acc_dict.get(k, [])
            l.append(v)
            self.acc_dict[k] = l

    def log_iter_stats(self, iter_data, cnt, batch_idx, image_dir=None, wandb_enabled=False, plot=False):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return

        if batch_idx % self.args.testing_config.log_every == 0:
            logger.info("Iter: {}, Log batch {} stats in FeatureExtractMeter".format(cnt, batch_idx))
        self.update_acc_dict(iter_data, cnt)

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

        