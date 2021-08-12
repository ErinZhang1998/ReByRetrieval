import yaml 
# from optparse import OptionParser
import argparse
import torch
import wandb
import os 

from models.build import build_model
import utils.multiprocessing as mpu
import utils.utils as uu
import train
import utils.distributed as du

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", dest="config_file")
parser.add_argument("--only_test", dest="only_test", action='store_true')
parser.add_argument("--only_test_epoch", dest="only_test_epoch", type=int, default=-1)
parser.add_argument(
        "--init_method",
        dest="init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
parser.add_argument("--model_path", dest="model_path", default='')
parser.add_argument("--experiment_save_dir", dest="experiment_save_dir", default='')
parser.add_argument("--experiment_save_dir_default", dest="experiment_save_dir_default", default='')

parser.add_argument("--testing_scene_dir", dest="testing_scene_dir", default='')


def fill_in_args_from_terminal(args, options):
    if options.testing_scene_dir != '':
        args.files.testing_scene_dir = options.testing_scene_dir
    if options.experiment_save_dir != '':
        args.experiment_save_dir = options.experiment_save_dir
    if options.experiment_save_dir_default != '':
        args.experiment_save_dir_default = options.experiment_save_dir_default
    
    if options.model_path != '':
        args.model_config.model_path = options.model_path
        args.training_config.epochs = int(options.model_path.split('/')[-1].split('.')[0])
    
    if options.only_test_epoch > 0:
        args.training_config.epochs = options.only_test_epoch
    if options.only_test:
        args.training_config.train = False

    return args

def main():
    # (options, args) = parser.parse_args()
    options = parser.parse_args()
    f =  open(options.config_file)
    # f = open('configs/config_all_occlusion.yaml')
    args_dict = yaml.safe_load(f)
    default_args_dict = yaml.safe_load(open('configs/default.yaml'))
    args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)
    args = uu.Struct(args_dict_filled)

    args = fill_in_args_from_terminal(args, options)

    if args.num_gpus > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=args.num_gpus,
            args=(
                args.num_gpus,
                train.train,
                options.init_method,
                0,
                1,
                "nccl",
                args,
            ),
            daemon=False,
        )
    else:
        train.train(args)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()