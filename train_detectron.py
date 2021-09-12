import numpy as np
import os, json, cv2, random
import yaml
import wandb
import argparse
import pickle

import matplotlib.pyplot as plt
import pycocotools.mask as coco_mask
import PIL

import utils.logging as logging
import utils.utils as uu
import utils.plot_image as uplot
from detectron_mapper import RetrievalMapper

# check pytorch installation: 
import torch, torchvision
assert torch.__version__.startswith("1.9")   
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.get_logger(__name__)

IDX_TO_NAME = {
    0: 'mug-cylinder',
    1: 'mug-square_handle',
    2: 'mug-tappered',
    3: 'bag-triangular',
    4: 'bag-rectangular',
    5: 'laptop-straight',
    6: 'laptop-slightly_slanted',
    7: 'laptop-slanted',
    8: 'jar-vase',
    9: 'jar-short_neck',
    10: 'can',
    11: 'camera-box',
    12: 'camera-protruded',
    13: 'bowl-shallow',
    14: 'bowl-deep',
    15: 'bottle-wine',
    16: 'weird',
    17: 'bottle-square-body',
    18: 'bottle-water',
    19: 'bottle-cylinder',
    20: 'bottle-pill',
    21: 'bottle-beer',
    22: 'bottle-flask',
    23: 'bottle-jar-like',
    24: 'bottle-round-body',
}

IDX_TO_NAME = {
    0: 'mug',
    1: 'bag',
    2: 'laptop',
    3: 'jar',
    4: 'can',
    5: 'camera',
    6: 'bowl',
    7: 'bottle',
}
NAME_TO_IDX = dict(zip(IDX_TO_NAME.values(), IDX_TO_NAME.keys()))

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", dest="config_file")
parser.add_argument("--experiment_save_dir", dest="experiment_save_dir")
parser.add_argument("--experiment_save_dir_default", dest="experiment_save_dir_default")
parser.add_argument("--init_method", dest="init_method", default="tcp://localhost:9999",type=str)
parser.add_argument("--resume", action="store_true", dest="resume")
parser.add_argument("--model_path", dest="model_path")


def get_data_detectron(args, split):
    if split == 'train':
        directory_list_file = args.detectron.train.directory_list_file
    else:
        directory_list_file = args.detectron.test.directory_list_file
    fh = open(directory_list_file, 'rb')
    dir_list = pickle.load(fh)
    fh.close()

    if args.dataset_config.only_load < 0:
        dir_list_load = dir_list
    else:
        dir_list_load = dir_list[:args.dataset_config.only_load]
    
    data_dict_all = []
    for dir_path in dir_list_load:
        data_dict_list = json.load(open(
            os.path.join(dir_path, 'detectron_annotations.json')
        ))
        for data_dict in data_dict_list:
            new_annos = []
            height = data_dict['height']
            width = data_dict['width']
            for ann in data_dict['annotations']:
                if ann['category_id'] is None:
                    ann['category_id'] = NAME_TO_IDX[ann['category_name']]
                
                ann['bbox_mode'] = BoxMode.XYWH_ABS
                is_polygon = ann['polygon']
                if not is_polygon:
                    rle = ann['segmentation']
                    segmentation = coco_mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
                else:
                    rles = coco_mask.frPyObjects(ann['segmentation'], height, width)
                    segmentation = coco_mask.merge(rles)
                ann['segmentation'] = segmentation
                new_annos += [ann]
            data_dict['annotations'] = new_annos 
            data_dict_all += [data_dict]

    return data_dict_all


def setup(options, args):
    if comm.is_main_process():
        if args.wandb_detectron.enable:
            wandb.login()
            wandb.init(
                project=args.wandb_detectron.wandb_project_name, 
                entity=args.wandb_detectron.wandb_project_entity, 
                config=args.obj_dict,
            )
        
    wandb_enabled = args.wandb_detectron.enable and not wandb.run is None
    if wandb_enabled:
        wandb_run_name = wandb.run.name 
    else:
        wandb_run_name = uu.get_timestamp()

    if options.experiment_save_dir is None:
        if comm.is_main_process():
            uu.create_dir(options.experiment_save_dir_default)
        experiment_save_dir = os.path.join(options.experiment_save_dir_default, wandb_run_name)
    else:
        experiment_save_dir = options.experiment_save_dir
    if comm.is_main_process():
        uu.create_dir(experiment_save_dir)
        uu.create_dir(os.path.join(experiment_save_dir, 'saved_images'))
        logging.setup_logging(log_to_file=args.log_to_file, experiment_dir=experiment_save_dir)
    
    logger.info("cfg.OUTPUT_DIR: {}".format(experiment_save_dir))
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (args.detectron.train.dataset_name,)
    cfg.DATASETS.TEST = (args.detectron.test.dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = args.detectron.dataloader.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    
    if options.model_path is not None:
        cfg.MODEL.WEIGHTS = options.model_path
    else:
        last_checkpoint_file = os.path.join(experiment_save_dir, 'last_checkpoint')
        if os.path.exists(last_checkpoint_file) and options.resume:
            ckpt_fh = open(last_checkpoint_file, 'r')
            ckpt_lines = ckpt_fh.readlines()
            ckpt_path = os.path.join(experiment_save_dir, ckpt_lines[0])
            if os.path.exists(ckpt_path):
                cfg.MODEL.WEIGHTS = ckpt_path
    
    cfg.SOLVER.IMS_PER_BATCH = args.detectron.solver.ims_per_batch
    cfg.SOLVER.BASE_LR = args.detectron.solver.base_lr   # pick a good LR
    cfg.SOLVER.MAX_ITER = args.detectron.solver.max_iter 
    cfg.SOLVER.CHECKPOINT_PERIOD = args.detectron.solver.checkpoint_period   
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(IDX_TO_NAME)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.TEST.EVAL_PERIOD = args.detectron.test.eval_period
    cfg.OUTPUT_DIR = experiment_save_dir
    cfg.WANDB_ENABLED = wandb_enabled

    return cfg


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=RetrievalMapper(cfg, is_train=False))
        evaluator = COCOEvaluator(
            dataset_name, 
            output_dir=os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]

    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg, mapper=RetrievalMapper(cfg, is_train=True))
    
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            if comm.is_main_process() and iteration % 1000 == 0:
                    
                selected_idx = np.random.choice(len(data))
                fig, axs = plt.subplots(1, 2, figsize=(30,20))
                image_PIL = torchvision.transforms.ToPILImage()(data[selected_idx]['image'])
                mask_PIL = torchvision.transforms.ToPILImage()(data[selected_idx]['instances'].gt_masks.tensor[0].float())
                background = PIL.Image.new("RGB", image_PIL.size, 0)
                masked_image = PIL.Image.composite(image_PIL, background, mask_PIL)
                axs[0].imshow(np.asarray(image_PIL))
                axs[1].imshow(np.asarray(masked_image))

                image_name = data[selected_idx]['image_id']

                if cfg.WANDB_ENABLED:
                    final_img = uplot.plt_to_image(fig)
                    log_key = '{}/{}'.format('train_image', image_name)
                    wandb.log({log_key: wandb.Image(final_img)}, step=iteration)
                else:
                    image_path = os.path.join(cfg.OUTPUT_DIR, 'saved_images', "iter_{}_{}.png".format(iteration, image_name))
                    plt.savefig(image_path)
                plt.close()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # import pdb; pdb.set_trace()
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                test_results = do_test(cfg, model)
                if comm.is_main_process():
                    for k,v in test_results.items():
                        if type(v) is not dict:
                            storage.put_scalar(f"test/{k}", v, smoothing_hint=False)
                        else:
                            for ki,vi in v.items():
                                storage.put_scalar(f"test/{k}_{ki}", vi, smoothing_hint=False)

                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def main(options, args):
    cfg = setup(options, args)
    model = build_model(cfg)
    distributed = comm.get_world_size() > 1

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    for split in ['train', 'test']:
        if split == 'train':
            DatasetCatalog.register(args.detectron.train.dataset_name, lambda split = split : get_data_detectron(args, split))
            MetadataCatalog.get(args.detectron.train.dataset_name).thing_classes = list(IDX_TO_NAME.values())
        else:
            DatasetCatalog.register(args.detectron.test.dataset_name, lambda split = split : get_data_detectron(args, split))
            MetadataCatalog.get(args.detectron.test.dataset_name).thing_classes = list(IDX_TO_NAME.values())
    
    do_train(cfg, model, resume=options.resume)

if __name__ == '__main__':
    options = parser.parse_args()
    f =  open(options.config_file)
    args_dict = yaml.safe_load(f)
    config_args = uu.Struct(args_dict)

    launch(
        main,
        config_args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=options.init_method,
        args=(options,config_args),
    )