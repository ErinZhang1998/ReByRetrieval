import yaml 
from optparse import OptionParser
import torch
import wandb
import os 

from models.build import build_model
import utils.multiprocessing as mpu
import utils.utils as uu
import train
import utils.distributed as du

parser = OptionParser()
parser.add_option("--config_file", dest="config_file")
parser.add_option("--only_test", dest="only_test", action='store_true')
parser.add_option("--only_test_epoch", dest="only_test_epoch", type=int, default=-1)
parser.add_option(
        "--init_method",
        dest="init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
parser.add_option("--model_path", dest="model_path", default='')


def main():
    (options, args) = parser.parse_args()
    f =  open(options.config_file)
    # f = open('configs/config_all_occlusion.yaml')
    args_dict = yaml.safe_load(f)
    default_args_dict = yaml.safe_load(open('configs/default.yaml'))
    args_dict_filled = uu.fill_in_args_from_default(args_dict, default_args_dict)
    args = uu.Struct(args_dict_filled)

    if options.model_path != '':
        args.model_config.model_path = options.model_path
    if options.only_test_epoch > 0:
        args.training_config.epochs = options.only_test_epoch
    if options.only_test:
        args.training_config.train = False

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