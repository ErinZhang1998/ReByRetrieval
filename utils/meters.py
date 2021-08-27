import torch
import numpy as np 
import wandb
import os 
import utils.transforms as utrans
import torchvision
import utils.plot_image as uplot 
import utils.distributed as du
import torch.distributed as dist
from sklearn.metrics import top_k_accuracy_score

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
        self.num_classes = -1

    def reset(self):
        for k in self.contrastive_loss_dict.keys():
            self.contrastive_loss_dict[k] = (0.0, 0.0)

        for k in self.acc_dict.keys():
            self.acc_dict[k] = []

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
                raise
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
    
    def update_acc_dict(self, iter_data):
        for k,v in iter_data.items():
            if k.startswith('contrastive_'):
                loss_acc, count = self.contrastive_loss_dict.get(k, (0.0, 0.0))
                loss_acc += v[0]
                count += v[1]
                self.contrastive_loss_dict[k] = (loss_acc, count)
            else:
                l = self.acc_dict.get(k, [])
                l.append(v)
                self.acc_dict[k] = l
    
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
        self.update_acc_dict(iter_data)
    
    def calculate_top_k_accuracy_score(self, loss_key, class_pred, label):

        class_pred = class_pred.numpy()
        gt_label = label.numpy().astype(int)
        assert self.num_classes > 0
        labels = np.arange(self.num_classes)

        acc_top_1 = top_k_accuracy_score(gt_label, class_pred, k=1, labels=labels)
        acc_top_5 = top_k_accuracy_score(gt_label, class_pred, k=5, labels=labels)
        acc_top_10 = top_k_accuracy_score(gt_label, class_pred, k=10, labels=labels)

        acc_top_k_dict = {
            f'test/{loss_key} acc@1' : acc_top_1,
            f'test/{loss_key} acc@5' : acc_top_5,
            f'test/{loss_key} acc@10' : acc_top_10,
        }
        return acc_top_k_dict
    
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
        for key in self.acc_dict.keys():
            if key == 'image':
                continue
            value = torch.cat(self.acc_dict[key], dim=0) #(batch_size, x)
            value = value.numpy()
            fname = os.path.join(prediction_dir, f'{epoch}_{key}.npy')
            np.save(fname, value)
    
    def finalize_metrics(self, epoch, cnt, prediction_dir):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return 
        
        save_prediction = epoch == self.args.training_config.epochs or epoch % self.args.testing_config.save_prediction_every == 0 or (not self.args.training_config.train)
        args = self.args
        logger.info('Validate Epoch: {} , Iteration: {}'.format(epoch, cnt))

        wandb_dict = {}
        return_keys = self.args.model_config.model_return
        if 'scale_pred' in return_keys:
            scale_pred = torch.cat(self.acc_dict['scale_pred'], dim=0)
            scale = torch.cat(self.acc_dict['scale'], dim=0)
            scale_loss = loss.get_loss_func(args.loss.scale_pred_fn)(scale_pred, scale) * args.loss.lambda_scale_pred
            scale_loss = scale_loss.item()
            logger.info('\tscale_loss={:.6f}'.format(scale_loss))
            wandb_dict['test/scale_loss'] = scale_loss

        if 'center_pred' in return_keys:
            center_pred = torch.cat(self.acc_dict['center_pred'], dim=0)
            center = torch.cat(self.acc_dict['center'], dim=0)
            center_loss = loss.get_loss_func(args.loss.center_pred_fn)(center_pred, center) * args.loss.lambda_center_pred
            logger.info('\tcenter_loss={:.6f}'.format(center_loss.item()))
            wandb_dict['test/center_loss'] = center_loss.item()
        
        if 'class_pred' in return_keys:
            class_pred = torch.cat(self.acc_dict['class_pred'], dim=0)
            class_type = self.args.model_config.class_type
            class_gt = torch.cat(self.acc_dict[class_type], dim=0)
            class_loss = loss.get_loss_func(args.loss.class_pred_fn)(class_pred, class_gt.view(-1,).long()) * args.loss.lambda_class_pred
            logger.info('\tclass_loss={:.6f} ({})'.format(class_loss.item(), class_type))
            wandb_dict['test/class_loss_{}'.format(class_type)] = class_loss.item()

            acc_top_k_dict = self.calculate_top_k_accuracy_score(f'class_{class_type}', class_pred, class_gt.view(-1).long())
            wandb_dict.update(acc_top_k_dict)
            for k,v in acc_top_k_dict.items():
                logger.info('\t{} = {:.6f}'.format(k, v))

        if 'img_embed' in return_keys:
            pred_val = torch.cat(self.acc_dict['img_embed'], dim=0)

            loss_value, count = self.contrastive_loss_dict['contrastive_obj_category']
            loss_value = loss_value / count
            logger.info('\tcontrastive_obj_category={:.6f}'.format(loss_value * self.args.loss.lambda_obj_category))

            loss_value, count = self.contrastive_loss_dict['contrastive_obj_id']
            loss_value = loss_value / count
            logger.info('\tcontrastive_obj_id={:.6f}'.format(loss_value * self.args.loss.lambda_obj_id))

            gt_val = torch.cat(self.acc_dict['obj_category'], dim=0)

            mapk_dict = self.calculate_mapk('obj_category', pred_val.numpy(), gt_val.numpy())
            wandb_dict.update(mapk_dict)
            for k,v in mapk_dict.items():
                logger.info('\t{} = {:.6f}'.format(k, v))
            
            gt_val = torch.cat(self.acc_dict['obj_id'], dim=0)
            mapk_dict = self.calculate_mapk('obj_id', pred_val.numpy(), gt_val.numpy())
            wandb_dict.update(mapk_dict)
            for k,v in mapk_dict.items():
                logger.info('\t{} = {:.6f}'.format(k, v))

        if self.args.wandb.enable and not wandb.run is None:
            wandb.log(wandb_dict, step=cnt)
        if save_prediction:
            self.save_prediction(epoch, cnt, prediction_dir)

class FeatureExtractMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, args):
        self.args = args
        self.acc_dict = dict()
        self.contrastive_loss_dict = dict()
        self.num_classes = -1

    def reset(self):
        for k in self.contrastive_loss_dict.keys():
            self.contrastive_loss_dict[k] = (0.0, 0.0)

        for k in self.acc_dict.keys():
            self.acc_dict[k] = []

    def update_acc_dict(self, iter_data):
        for k,v in iter_data.items():
            if k.startswith('contrastive_'):
                loss_acc, count = self.contrastive_loss_dict.get(k, (0.0, 0.0))
                loss_acc += v[0]
                count += v[1]
                self.contrastive_loss_dict[k] = (loss_acc, count)
            else:
                l = self.acc_dict.get(k, [])
                l.append(v)
                self.acc_dict[k] = l
    
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
        self.update_acc_dict(iter_data)

    def save_prediction(self, epoch, cnt, prediction_dir):
        for key in self.acc_dict.keys():
            if key == 'image':
                continue
            value = torch.cat(self.acc_dict[key], dim=0) #(batch_size, x)
            value = value.numpy()
            fname = os.path.join(prediction_dir, f'{epoch}_{key}.npy')
            np.save(fname, value)
    
    def finalize_metrics(self, epoch, cnt, prediction_dir):
        if not du.is_master_proc(num_gpus=self.args.num_gpus):
            return 
        
        save_prediction = epoch == self.args.training_config.epochs or epoch % self.args.testing_config.save_prediction_every == 0 or (not self.args.training_config.train)

        if save_prediction:
            self.save_prediction(epoch, cnt, prediction_dir)