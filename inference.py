import os 
import yaml
import argparse
import numpy as np 
import pandas as pd
import pickle
import torch
import shutil
import tqdm

import utils.perch_utils as p_utils
import utils.qualitative_utils as q_utils
import utils.datagen_utils as datagen_utils
import utils.blender_proc_utils as bp_utils
import utils.transforms as utrans
import utils.utils as uu

import incat_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", dest="config_yaml")
parser.add_argument("--output_retrieval_results", action="store_true", dest="output_retrieval_results")
parser.add_argument("--save_pred_size", action="store_true", dest="save_pred_size")


def sample_id_to_parts(sample_id):
    scene_num, image_id, category_id = sample_id.split('-')
    scene_num = int(scene_num)
    image_id = int(image_id)
    category_id = int(category_id)
    return scene_num, image_id, category_id

def pariwise_distances(X, Y, squared=False):
    XX = torch.matmul(X, torch.transpose(X, 0, 1)) #mxm
    YY = torch.matmul(Y, torch.transpose(Y, 0, 1)) #nxn
    YX = torch.matmul(Y, torch.transpose(X, 0, 1)) #nxm
    
    XX = torch.diagonal(XX)
    XX = torch.unsqueeze(XX, 0) #1xm
    
    YY = torch.diagonal(YY)
    YY = torch.unsqueeze(YY, 1) #nx1
    
    distances = XX - 2.0 * YX + YY #nxm
    distances = torch.max(distances, torch.zeros_like(distances))
    if not squared:
        mask = torch.isclose(distances, torch.zeros_like(distances), rtol=0).float()
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
    
    return distances.T #mxn

def torchify(feat):
    feat = torch.FloatTensor(feat)
    feat = feat.cuda()
    return feat

def get_features(save_dir, epoch, fname_template = '{}_embedding.npy'):
    prediction_dir = os.path.join(save_dir, 'predictions')
    features = np.load(os.path.join(prediction_dir, fname_template.format(epoch)))
    return features

def get_sample_ids(save_dir, epoch, fname_template = '{}_sample_id.npy'):
    prediction_dir = os.path.join(save_dir, 'predictions')
    sample_id = np.load(os.path.join(prediction_dir, fname_template.format(epoch))) 
    sample_id_res = []
    for L in sample_id:
        sample_id_res.append('-'.join([str(int(item)) for item in L]))
    return sample_id_res

def get_all_info(save_dir, epoch, df_all):
    object_ids = get_features(save_dir, epoch, fname_template = '{}_obj_id.npy').reshape(-1,)
    object_category = get_features(save_dir, epoch, fname_template = '{}_obj_category.npy').reshape(-1,)
    model_id_idx = get_features(save_dir, epoch, fname_template = '{}_shapenet_model_id.npy').reshape(-1,)
    model_id = df_all.iloc[list(model_id_idx),:]['ShapeNetModelId'].to_numpy()
    
    sample_ids = get_sample_ids(save_dir, epoch)
    sample_ids = np.asarray(sample_ids).reshape(-1,)
    
    d = {
        'self_defined_category': object_ids.astype(int), 
        'shapenet_category': object_category.astype(int),
        'sample_id' : sample_ids,
        'model_id' : model_id,
    }
    df = pd.DataFrame(data=d)
    return df
    
def uniform_distribution(save_dir, epoch, df_all, key = 'self_defined_category'):
    df = get_all_info(save_dir, epoch, df_all)    
    min_num = df[key].value_counts().min()
    selected_idx_list = []
    for idx in df[key].unique():
        selected_idx = df[df.self_defined_category == idx].sample(min_num).index.to_numpy()
        selected_idx_list.append(selected_idx)
    
    selected_idx = np.hstack(selected_idx_list)
    return df, selected_idx

def all_to_old_annotaiton_format(storage_root, perch_model_dir, yaml_file_root_dir, scene_save_dir, df_all):
    for scene_sub_dir in os.listdir(scene_save_dir):
        if not scene_sub_dir.startswith('scene_'):
            continue 
        one_scene_dir = os.path.join(scene_save_dir, scene_sub_dir)
        print("To old annotations: ", one_scene_dir)
        bp_utils.to_old_annotaiton_format(
            perch_model_dir, 
            yaml_file_root_dir, 
            df_all, 
            one_scene_dir,
            storage_root = storage_root,
        )

def output_retrieval_results(args):
    df_all = pd.read_csv(os.path.join(args.file_root, args.all_csv_file))
    
    # bb_max_min_info = pickle.load(open(args.bb_max_min_info, 'rb'))
    # query_train_or_test = args.query.data_dir.split('/')[-1]
    target_train_or_test = args.target.data_dir.split('/')[-1]
    experiment_name = f'{args.result_name}-{target_train_or_test}-{args.target.epoch}-{args.query.epoch}'

    # pred_scale = get_features(args.query.save_dir, args.query.epoch, fname_template = '{}_scale_pred.npy').reshape(-1,)
    # df_query = get_all_info(args.query.save_dir, args.query.epoch, df_all)
    # df_target = get_all_info(args.target.save_dir, args.target.epoch, df_all)
    # df_target, selected_idx = uniform_distribution(args.target.save_dir, args.target.epoch, df_all, key = 'self_defined_category')

    query_feats = get_features(args.query.save_dir, args.query.epoch, fname_template = '{}_img_embed.npy')
    target_feats = get_features(args.target.save_dir, args.target.epoch, fname_template = '{}_img_embed.npy')
    # target_feats = target_feats[selected_idx]
    query_feats = torchify(query_feats)
    target_feats = torchify(target_feats)

    pairwise_dist = pariwise_distances(query_feats, target_feats, squared=False)
    sorted_dist, arg_sorted_dist = torch.sort(pairwise_dist, dim=1)

    # # query_sample_ids = df_query.sample_id.to_numpy()
    # target_sample_ids = df_target.sample_id.to_numpy()

    arg_sorted_dist = arg_sorted_dist.cpu().numpy()
    # selected_target_sample_id = np.asarray(target_sample_ids)[arg_sorted_dist]

    retrieval_result_dir = os.path.join(args.query.save_dir, experiment_name)
    if not os.path.exists(retrieval_result_dir):
        os.mkdir(retrieval_result_dir)
    
    fname = os.path.join(retrieval_result_dir, 'result.npy')
    np.save(fname, arg_sorted_dist)

def save_pred_size(args):

    df_all = pd.read_csv(os.path.join(args.file_root, args.all_csv_file))
    
    bb_max_min_info = pickle.load(open(args.bb_max_min_info, 'rb'))
    query_train_or_test = args.query.data_dir.split('/')[-1]
    target_train_or_test = args.target.data_dir.split('/')[-1]
    experiment_name = f'{args.result_name}-{target_train_or_test}-{args.target.epoch}-{args.query.epoch}'

    pred_scale = get_features(args.query.save_dir, args.query.epoch, fname_template = '{}_scale_pred.npy').reshape(-1,)
    df_query = get_all_info(args.query.save_dir, args.query.epoch, df_all)
    df_target = get_all_info(args.target.save_dir, args.target.epoch, df_all)

    query_sample_ids = df_query.sample_id.to_numpy()
    target_sample_ids = df_target.sample_id.to_numpy()

    fname = os.path.join(args.query.save_dir, experiment_name, 'result.npy')
    arg_sorted_dist = np.load(fname)
    print(arg_sorted_dist.shape)
    selected_target_sample_id = np.asarray(target_sample_ids)[arg_sorted_dist]

    all_pred_sizes = []
    all_actual_sizes = []
    for query_idx in tqdm.tqdm(range(len(query_sample_ids))):
        sample_id = query_sample_ids[query_idx]
        target_sample_ids = selected_target_sample_id[query_idx]

        query_scene, query_image, category_id1 = sample_id_to_parts(sample_id)
        query_annotation_path = os.path.join(
            args.query.data_dir,
            f'scene_{query_scene:06}',
            'image_annotations',
            args.new_fname_template.format(query_image),
        )            
        json_obj_2 = p_utils.COCOSelf(query_annotation_path)
        query_ann = json_obj_2.category_id_to_ann[category_id1]
        all_actual_sizes.append(query_ann['actual_size'])

        pred_sizes = []
        for prediction_idx in range(10):
            target_sample_id = target_sample_ids[prediction_idx] 
            target_scene, target_image, category_id2 = sample_id_to_parts(target_sample_id)

            model_2_annotation_path = os.path.join(args.target.data_dir, f'scene_{target_scene:06}', 'annotations.json')
            json_obj_2 = p_utils.COCOSelf(model_2_annotation_path)
            target_ann = json_obj_2.category_id_to_ann[category_id2]

            synset_id = target_ann['synset_id']
            model_id = target_ann['model_id']
            bb_max, bb_min = bb_max_min_info[(synset_id, model_id)]
            scale = pred_scale[query_idx]
            pred_size = (bb_max - bb_min) * np.array([scale] * 3)
            pred_sizes.append(pred_size)
        
        all_pred_sizes.append(pred_sizes)
    
    retrieval_result_dir = os.path.join(args.query.save_dir, experiment_name)
    if not os.path.exists(retrieval_result_dir):
        os.mkdir(retrieval_result_dir)
    
    fname = os.path.join(retrieval_result_dir, 'pred_size.npy')
    np.save(fname, np.asarray(all_pred_sizes))

    fname = os.path.join(retrieval_result_dir, 'actual_size.npy')
    np.save(fname, np.asarray(all_actual_sizes))


def main(args):
    df_all = pd.read_csv(os.path.join(args.file_root, args.all_csv_file))
    
    bb_max_min_info = pickle.load(open(args.bb_max_min_info, 'rb'))
    query_train_or_test = args.query.data_dir.split('/')[-1]
    target_train_or_test = args.target.data_dir.split('/')[-1]
    experiment_name = f'{args.result_name}-{target_train_or_test}-{args.target.epoch}-{args.query.epoch}'

    pred_scale = get_features(args.query.save_dir, args.query.epoch, fname_template = '{}_scale_pred.npy').reshape(-1,)
    df_query = get_all_info(args.query.save_dir, args.query.epoch, df_all)
    df_target, selected_idx = uniform_distribution(args.target.save_dir, args.target.epoch, df_all, key = 'self_defined_category')

    query_feats = get_features(args.query.save_dir, args.query.epoch, fname_template = '{}_img_embed.npy')
    target_feats = get_features(args.target.save_dir, args.target.epoch, fname_template = '{}_img_embed.npy')
    target_feats = target_feats[selected_idx]
    query_feats = torchify(query_feats)
    target_feats = torchify(target_feats)

    pairwise_dist = pariwise_distances(query_feats, target_feats, squared=False)
    sorted_dist, arg_sorted_dist = torch.sort(pairwise_dist, dim=1)

    query_sample_ids = df_query.sample_id.to_numpy()
    target_sample_ids = df_target.sample_id.to_numpy()

    arg_sorted_dist = arg_sorted_dist.cpu().numpy()
    selected_target_sample_id = np.asarray(target_sample_ids)[arg_sorted_dist]
    # import pdb; pdb.set_trace()

    # for query_idx,(sample_id, target_sample_ids) in enumerate(zip(query_sample_ids, selected_target_sample_id)):
    for query_idx in tqdm.tqdm(range(len(query_sample_ids))):
        sample_id = query_sample_ids[query_idx]
        target_sample_ids = selected_target_sample_id[query_idx]
        
        for prediction_idx in range(5):
            target_sample_id = target_sample_ids[prediction_idx] 
            new_model_name = '{}__{}__{}'.format(sample_id, target_sample_id, prediction_idx)
        
            # print("\n", new_model_name, sample_id, target_sample_id)
        
            query_scene, query_image, category_id1 = sample_id_to_parts(sample_id)
            target_scene, target_image, category_id2 = sample_id_to_parts(target_sample_id)

            scene_sub_dir = f'scene_{query_scene:06}'
            original_anno_path = os.path.join(
                args.query.data_dir,
                scene_sub_dir,
                'image_annotations',
                args.new_fname_template.format(query_image),
            )
            new_anno_dir = os.path.join(
                args.query.data_dir,
                scene_sub_dir,
                experiment_name,
            )
            if not os.path.exists(new_anno_dir):
                os.mkdir(new_anno_dir) 
            new_anno_path = os.path.join(new_anno_dir, '{}.json'.format(new_model_name))

            model_2_annotation_path = os.path.join(args.target.data_dir, f'scene_{target_scene:06}', 'annotations.json')
            json_obj_2 = p_utils.COCOSelf(model_2_annotation_path)
            target_ann = json_obj_2.category_id_to_ann[category_id2]

            synset_id = target_ann['synset_id']
            model_id = target_ann['model_id']
            bb_max, bb_min = bb_max_min_info[(synset_id, model_id)]
            scale = pred_scale[query_idx]
            pred_size = (bb_max - bb_min) * np.array([scale] * 3)

            p_utils.paste_in_new_category_annotation_perch(
                original_anno_path=original_anno_path,
                new_anno_path=new_anno_path,
                category_id=category_id1,
                new_annotation=target_ann,
                new_model_name=new_model_name,
                model_save_dir=args.perch_model_dir,
                normalized_shapenet_model_dir=args.blender_proc_model_dir,
                turn_upright_before_scale=False,
                turn_upright_after_scale=True,
                save_new_model=True,
                new_actual_size=pred_size,
                new_scale=[scale]*3,
                over_write_new_anno_path=False,
            )
            

def run_to_old_annotation(args):
    df_all = pd.read_csv(os.path.join(args.file_root, args.all_csv_file))
    if args.target.run_to_old_annotation:
        all_to_old_annotaiton_format(args.storage_root, args.perch_model_dir, args.target.yaml_file_root_dir, args.target.data_dir, df_all)
    if args.query.run_to_old_annotation:
        all_to_old_annotaiton_format(args.storage_root, args.perch_model_dir, args.query.yaml_file_root_dir, args.query.data_dir, df_all)


def split_into_per_image_annotation(args):
    for scene_sub_dir in os.listdir(args.query.data_dir):
        if not scene_sub_dir.startswith('scene_'):
            continue 

        coco_anno_path = os.path.join(
            args.query.data_dir,
            scene_sub_dir,
            'annotations.json',
        )
        if not os.path.exists(coco_anno_path):
            print("WARNING! DOES NOT EXIST: ", coco_anno_path)
            continue
            
        new_fname_dir = os.path.join(args.query.data_dir, scene_sub_dir, 'image_annotations')
        if not os.path.exists(new_fname_dir):
            os.mkdir(new_fname_dir)
        
        print("Processing: ", coco_anno_path)
        p_utils.separate_annotation_into_images(
            args.query.model_dir,
            coco_anno_path, 
            new_fname_dir, 
            args.new_fname_template, 
            skip_image_ids = None,
            model_name_suffx_template = None,
            new_anno_model_save_root_dir = args.perch_model_dir,
        )


def clean(args):
    target_train_or_test = args.target.data_dir.split('/')[-1]
    experiment_name = experiment_name = f'{args.result_name}-{target_train_or_test}-{args.target.epoch}-{args.query.epoch}'
    for scene_sub_dir in os.listdir(args.query.data_dir):
        if not scene_sub_dir.startswith('scene_'):
            continue 
        new_anno_dir = os.path.join(
            args.query.data_dir,
            scene_sub_dir,
            experiment_name,
        )
        if os.path.exists(new_anno_dir):
            print("Remove: ", new_anno_dir)
            shutil.rmtree(new_anno_dir)

def run(args):
    run_to_old_annotation(args)

    if args.split_into_per_image_annotation:
        split_into_per_image_annotation(args)
    
    if args.clean:
        clean(args)

    if args.run_main:
        main(args)


if __name__ == '__main__':

    options = parser.parse_args()
    cfg =  open(options.config_yaml, 'r')
    args_dict = yaml.safe_load(cfg)
    args = uu.Struct(args_dict)
    cfg.close()

    if options.output_retrieval_results:
        output_retrieval_results(args)
    elif options.save_pred_size:
        print("hello")
        save_pred_size(args)
    else:
        run(args)
    

