import os
import yaml
import argparse

import utils.utils as uu
import utils.perch_utils as p_utils 


parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", dest="config_yaml")


if __name__ == '__main__':
    options = parser.parse_args()
    cfg =  open(options.config_yaml, 'r')
    args_dict = yaml.safe_load(cfg)
    args = uu.Struct(args_dict)
    cfg.close()

    experiment_name = p_utils.get_experiment_names(args)
    annotation_list_path = [
        os.path.join(
            args.perch_pickle_file_dir,
            args.perch_pickle_file_template.format(idx+1),
        ) for idx in range(args.number_of_perch_runs)
    ]
    perch_output_dir = os.path.join(args.perch_output_root_dir, experiment_name)

    result = p_utils.from_annotation_list_path_to_model_dict_list(
        perch_output_dir, 
        annotation_list_path, 
        args.perch_root_dir,
    )