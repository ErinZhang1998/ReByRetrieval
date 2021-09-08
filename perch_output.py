import os
import yaml
import pickle
import argparse
import pandas as pd 

import utils.utils as uu
import utils.perch_utils as p_utils 


parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", dest="config_yaml")


def compile_add_s(df_names_list, query_sample_ids):
    add_s_dict = {}
    index = []
    for sample_id in query_sample_ids:
        row = {}
        for i in range(len(df_names_list)):
            name,df = df_names_list[i]

            df_selected = df[df['sample_id'] == sample_id]
            df_selected = df_selected[df_selected['add-s'] >= 0]
            
            if len(df_selected) < 1:
                row[name] = np.nan
            else:
                min_add_s_idx = df_selected['add-s'].idxmin()
                add_s = df.iloc[min_add_s_idx]['add-s']
                row[name] = add_s
            
        
        add_s_dict[sample_id] = row
        index.append(sample_id)
    
    df_add_s = pd.DataFrame.from_dict(add_s_dict, orient='index')
    return df_add_s

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

    with open(os.path.join(args.perch_output_root_dir, f'{experiment_name}_model_dict_list.pkl'), 'wb+') as fh:
        pickle.dump([], fh)
    
    result = p_utils.from_annotation_list_path_to_model_dict_list(
        perch_output_dir, 
        annotation_list_path, 
        args.perch_root_dir,
    )

    csv_file = os.path.join(args.perch_output_root_dir, f'{experiment_name}_result.csv')
    result['df'].to_csv(csv_file, index=False)

    with open(os.path.join(args.perch_output_root_dir, f'{experiment_name}_model_dict_list.pkl'), 'wb+') as fh:
        pickle.dump(result['model_dict_list'], fh)
    
    with open(os.path.join(args.perch_output_root_dir, f'{experiment_name}_anno_list.pkl'), 'wb+') as fh:
        pickle.dump(result['anno_list'], fh)
    
    with open(os.path.join(args.perch_output_root_dir, f'{experiment_name}_doesnt_exist.pkl'), 'wb+') as fh:
        pickle.dump(result['doesnt_exist'], fh)