import numpy as np
import os, json, cv2, random
import yaml
import wandb
import argparse
import pandas as pd

import matplotlib.pyplot as plt
import pycocotools.mask as coco_mask


import utils.utils as uu
import incat_dataset
import utils.perch_utils as p_utils
import utils.blender_proc_utils as bp_utils

# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   

# Some basic setup:
# Setup detectron2 logger
import detectron2
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
from detectron2.data import build_detection_test_loader

idx_to_name = {
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

idx_to_name = {
    0: 'mug',
    1: 'bag',
    2: 'laptop',
    3: 'jar',
    4: 'can',
    5: 'camera',
    6: 'bowl',
    7: 'bottle',
}

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", dest="config_file")
parser.add_argument("--experiment_save_dir", dest="experiment_save_dir")
parser.add_argument("--experiment_save_dir_default", dest="experiment_save_dir_default")



def load_annotations_detectron(args, split, df, one_scene_dir):
    scene_num = int(one_scene_dir.split('/')[-1].split('_')[-1])

    yaml_file_prefix = '_'.join(one_scene_dir.split('/')[-2:])
    if split == 'train':
        yaml_file = os.path.join(args.files.training_yaml_file_dir, '{}.yaml'.format(yaml_file_prefix))
    else:
        yaml_file = os.path.join(args.files.testing_yaml_file_dir, '{}.yaml'.format(yaml_file_prefix))

    yaml_file_obj = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
    datagen_yaml = bp_utils.from_yaml_to_object_information(yaml_file_obj, df)
    coco_fname = os.path.join(one_scene_dir, 'coco_data', 'coco_annotations.json')
    coco = p_utils.COCOSelf(coco_fname)

    data_dict_list = []
    for image_id, image_ann in coco.image_id_to_ann.items():        
        image_id_across_dataset = '-'.join([str(scene_num), str(image_id)])
        image_file_name_full = os.path.join(one_scene_dir, image_ann['file_name'])

        annotations = []
        for category_id, ann in coco.image_id_to_category_id_to_ann[image_id].items():
            assert image_id == ann['image_id']
            
            if category_id == 0:
                continue
            if ann['area'] < args.dataset_config.ignore_num_pixels:
                continue
            if category_id not in datagen_yaml:
                continue
            if 'scale' not in datagen_yaml[category_id]:
                continue
            rle = ann['segmentation']
            annotation = {
                'bbox' : ann['bbox'],
                'bbox_mode' : BoxMode.XYWH_ABS,
                'category_id' : int(datagen_yaml[category_id]['obj_cat']),
                'segmentation' : coco_mask.frPyObjects(rle, rle['size'][0], rle['size'][1]),
            }
            annotations += [annotation]
        
        image_sample = {
            'file_name' : image_file_name_full,
            'width' : image_ann['width'],
            'height' : image_ann['height'],
            'image_id' : image_id_across_dataset,
            'annotations' : annotations,
        }
        data_dict_list += [image_sample]
    return data_dict_list

                
def get_data_detectron(args, split):
    df = pd.read_csv(args.files.csv_file_path)
    
    if split == 'train':
        scene_dir = args.files.training_scene_dir
    elif split == 'test':
        scene_dir = args.files.testing_scene_dir
    else:
        raise

    print(scene_dir)
    dir_list = uu.data_dir_list(
        scene_dir, 
        must_contain_file = ['0.hdf5', 'coco_data/coco_annotations.json']
    )

    if args.dataset_config.only_load < 0:
        dir_list_load = dir_list
    else:
        dir_list_load = dir_list[:args.dataset_config.only_load]
    
    data_dict_lists = []
    for dir_path in dir_list_load:
        data_dict_list = load_annotations_detectron(args, split, df, dir_path)
        data_dict_lists += data_dict_list
    
    return data_dict_lists


def main(options, args):
    for split in ["train", "test"]:
        DatasetCatalog.register(
            "blender_proc_dataset_" + split, 
            lambda split = split: get_data_detectron(args, split),
        )
        MetadataCatalog.get("blender_proc_dataset_" + split).thing_classes = list(idx_to_name.values())
    
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
        uu.create_dir(options.experiment_save_dir_default)
        this_experiment_dir = os.path.join(options.experiment_save_dir_default, wandb_run_name)
    else:
        this_experiment_dir = options.experiment_save_dir
    uu.create_dir(this_experiment_dir)
    
    print("cfg.OUTPUT_DIR: ", this_experiment_dir)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("blender_proc_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(idx_to_name)  
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = this_experiment_dir

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("blender_proc_dataset_test", output_dir=this_experiment_dir)
    val_loader = build_detection_test_loader(cfg, "blender_proc_dataset_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`

if __name__ == '__main__':
    options = parser.parse_args()
    f =  open(options.config_file)
    args_dict = yaml.safe_load(f)
    default_args_dict = yaml.safe_load(open('configs/default.yaml'))
    args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)
    args = uu.Struct(args_dict_filled)

    main(options, args)