import os
import json
import copy
import shutil
import traceback
import numpy as np
import pandas as pd

from datagen_args import *
from utils.datagen_utils import *
from perch_scene import *
from utils.perch_utils import COCOAnnotation
from blender_proc_scene import BlenderProcScene

# bag, bottle, bowl, can, clock, jar, laptop, camera, mug
_ACTUAL_LIMIT_DICT = {
    0 : (0.15, 0.3),
    1 : (0.15, 0.3),
    2 : (0.08, 0.11),
    3 : (0.08, 0.11),
    4 : (0.08, 0.11),
    5 : (0.08, 0.11),
    6 : (0.08, 0.11),
    7 : (0.14, 0.2),
    8 : (0.08, 0.11),
    9 : (0.12, 0.16),
    10 : (0.12, 0.18),
    11 : (0.12, 0.16),
    12 : (0.15, 0.2),
    13 : (0.08, 0.14),
    14 : (0.08, 0.14),
    15 : (0.3, 0.35),
    16 : (0.1, 0.15),
    17 : (0.13, 0.16),
    18 : (0.13, 0.16),
    19 : (0.13, 0.16),
}

_ACTUAL_LIMIT_BLENDER_DICT = {
    0 : (0.15, 0.20),
    1 : (0.15, 0.20),
    2 : (0.10, 0.15),
    3 : (0.10, 0.15),
    4 : (0.10, 0.15),
    5 : (0.10, 0.15),
    6 : (0.10, 0.15),
    7 : (0.15, 0.20),
    8 : (0.10, 0.15),
    9 : (0.10, 0.15),
    10 : (0.10, 0.15),
    11 : (0.10, 0.15),
    12 : (0.15, 0.20),
    13 : (0.15, 0.20),
    14 : (0.15, 0.20),
    15 : (0.15, 0.20),
    16 : (0.10, 0.15),
    17 : (0.10, 0.15),
    18 : (0.10, 0.15),
    19 : (0.10, 0.15),
}

def read_perch_output_poses(fname):
    annotations = []
    f = open(fname, "r")
    lines = f.readlines()
    if len(lines) == 0:
        print("Invalid PERCH run : {}".format(len(lines)))
        return None
   
    for i in np.arange(0, len(lines), 13):
        location = list(map(float, lines[i+1].rstrip().split()[1:]))
        quaternion = list(map(float, lines[i+2].rstrip().split()[1:]))
        transform_matrix = np.zeros((4,4))
        preprocessing_transform_matrix = np.zeros((4,4))
        for l_t in range(4, 8) :
            transform_matrix[l_t - 4,:] = list(map(float, lines[i+l_t].rstrip().split()))
        for l_t in range(9, 13) :
            preprocessing_transform_matrix[l_t - 9,:] = list(map(float, lines[i+l_t].rstrip().split()))
        annotations.append({
                        'location' : [location[0], location[1], location[2]],
                        'quaternion_xyzw' : quaternion,
                        'model_name' : lines[i].rstrip(),
                        'transform_matrix' : transform_matrix,
                        'preprocessing_transform_matrix' : preprocessing_transform_matrix
                    })
    
    annotations_dict = {}
    for ann in annotations:
        annotations_dict[ann['model_name']] = ann
    
    return annotations_dict

def get_camera_locations(camera_annotation_file):
    coco_anno = json.load(open(camera_annotation_file))
    locations = []
    for image_ann in coco_anno['images']:
        locations.append(image_ann['pos'])
    return locations

'''
from_world_frame_annotations_to_perch_cam(gt_coco_anno['categories'][0]['position'], gt_coco_anno['categories'][0]['quat'], gt_coco_anno['images'][19])
from_world_frame_annotations_to_perch_cam(gt_coco_anno['categories'][1]['position'], gt_coco_anno['categories'][1]['quat'], gt_coco_anno['images'][19])

'''

def from_file(args, scene_num, output_poses_txt, json_annotation_file, camera_annotation_file):
    '''
    output_poses_txt: output from perch, per image base. 
        position
        quaternion_xyzw
    json_annotation_file: 
        images[0]: contains camera position, orientation of the input image (cam->world matrix, world->cam matrix)
        categories:
            model_name: matches with that used in perch, f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}'
            actual_size: size of the object to scale to 
    '''
    # output_poses_txt = args.perch_annotation_file
    # json_path = args.json_annotation_file
    pred_annotations_dict = read_perch_output_poses(output_poses_txt)
    coco_anno = COCOAnnotation(json_annotation_file)
    image_ann = coco_anno.get_ann('images', 0)
    selected_objects_i = []

    ### DEBUG
    # gt_coco_anno = json.load(open(camera_annotation_file))
    ###
    
    for category_id, category_ann in coco_anno.total_ann_dict['categories'].items(): 
        # if not category_id in [0,1,2]:
        #     continue   
        model_name = category_ann['name']
        if model_name not in pred_annotations_dict:
            continue 
        pred_perch_ann = pred_annotations_dict[model_name]
        position = pred_perch_ann['location']
        quaternion_xyzw = pred_perch_ann['quaternion_xyzw']
        
        
        position_world, quat_world = from_perch_cam_annotations_to_world_frame(position, quaternion_xyzw, image_ann)
        # 
        # [1:] because '0' at the front
        selected_dict = {
            'synsetId' : category_ann['synset_id'][1:],
            'catId' : category_ann['shapenet_category_id'],
            'ShapeNetModelId' : category_ann['model_id'],
            'objId' : category_ann['shapenet_object_id'],
            'size' : category_ann['actual_size'],
            'half_or_whole' : category_ann['half_or_whole'],
            'perch_rot_angle' : category_ann['perch_rot_angle'],
            'position' : position_world,
            'quaternion_xyzw' : quat_world,
            'size_xyz' : True,
        }
        selected_objects_i.append(selected_dict)
    
    # if args.predefined_camera_locations:
    #     camera_locations = get_camera_locations(camera_annotation_file)
    # else:
    #     camera_locations = None
    camera_locations = get_camera_locations(camera_annotation_file)

    scene_folder_path = None
    try:
        perch_scene = PerchScene(scene_num, selected_objects_i, args, camera_locations=camera_locations)
        scene_folder_path = perch_scene.scene_folder_path
        perch_scene.create_convex_decomposed_scene()
        perch_scene.create_camera_scene()
    except:
        print('##################################### GEN Error!')
        if scene_folder_path is not None:
            shutil.rmtree(scene_folder_path)
        traceback.print_exc()

'''

testing_set_2 _ scene_000016 _ rgb_00019

scene_dir: /raid/xiaoyuz1/perch/perch_balance/testing_set_2/scene_000016/
/raid/xiaoyuz1/perch/perch_balance/perch_output_2/testing_set_2_scene_000016_rgb_00019/output_poses.txt
'''
def run_all_images(args):
    scene_dir = args.scene_dir
    perch_dir = args.perch_dir 

    scene_base_dir = scene_dir.split("/")[-2]
    scene_name = scene_dir.split("/")[-1]
    camera_annotation_file = os.path.join(scene_dir, 'annotations.json')
    all_annotations = json.load(open(camera_annotation_file))
    
    scene_num = args.start_scene_idx
    for image_ann in all_annotations['images']:
        image_id = image_ann['id']
        rgb_name = image_ann['file_name'].split("/")[-1].split(".")[0]
        perch_name = '_'.join([scene_base_dir, scene_name, rgb_name])
        
        output_poses_txt = os.path.join(perch_dir, perch_name, 'output_poses.txt')
        json_annotation_file = os.path.join(scene_dir, 'annotations_{}.json'.format(image_id))

        from_file(args, scene_num, output_poses_txt, json_annotation_file, camera_annotation_file)

        scene_num += 1


def create_one_6d_scene(scene_num, selected_objects, args):
    # perch_scene = PerchScene(scene_num, selected_objects, args)
    scene_folder_path = None
    try:
        perch_scene = PerchScene(scene_num, selected_objects, args)
        scene_folder_path = perch_scene.scene_folder_path
        perch_scene.create_convex_decomposed_scene()
        perch_scene.create_camera_scene()
        # 
    except:
        print('##################################### GEN Error!')
        if scene_folder_path is not None:
            shutil.rmtree(scene_folder_path)
        print(selected_objects)
        traceback.print_exc()


def get_selected_objects(args):
    unit_x, unit_y = args.wall_unit_x / 2, args.wall_unit_y / 2
    axis_grid_x = np.linspace(-unit_x, unit_x, 3)
    axis_grid_y = np.linspace(-unit_y, unit_y, 3)
    x_pos, y_pos = np.meshgrid(axis_grid_x, axis_grid_y)
    LOCATION_GRID = np.hstack([x_pos.reshape(-1,1), y_pos.reshape(-1,1)])
    
    df = pd.read_csv(args.csv_file_path)

    actual_size_choices = {}
    for i in range(len(df)):
        if not args.blender_proc:
            size_low, size_high = _ACTUAL_LIMIT_DICT[df.iloc[i]['objId']]
        else:
            size_low, size_high = 0.15, 0.2
            #_ACTUAL_LIMIT_BLENDER_DICT[df.iloc[i]['objId']]
        # num_steps = (size_high - size_low) / 0.01 + 1
        actual_size_choices[i] = list(np.linspace(size_low, size_high, 5))

    # TO CREATE A MORE BALANCED DATASET 
    unique_object_ids = df['objId'].unique()
    selected_object_indices = []

    for scene_num in range(args.num_scenes):
        num_object = np.random.randint(args.min_num_objects, args.max_num_objects+1, 1)[0]
        object_object_ids = np.random.choice(unique_object_ids, num_object, replace=True)
        selected_object_idx = []
        for obj_id in object_object_ids:
            obj_id_df = df[df['objId'] == obj_id]
            idx_in_obj_id_df = np.random.randint(0, len(obj_id_df), 1)[0]
            selected_object_idx.append(obj_id_df.iloc[idx_in_obj_id_df].name)

        selected_object_indices.append(selected_object_idx)
        
        
    selected_objects = []
    category_list = []
    object_list = []

    for selected_indices in selected_object_indices:
        selected_objects_i = []
        actual_size_choices_i = copy.deepcopy(actual_size_choices)
        location_grid_choice = copy.deepcopy(LOCATION_GRID)
        
        for idx in selected_indices:
            if len(location_grid_choice) < 1:
                break 
            
            loc = np.random.choice(len(location_grid_choice))
            x,y = location_grid_choice[loc]
            location_grid_choice = np.delete(location_grid_choice, loc, 0)

            row = df.iloc[idx]
            cat_actual_size_choice = actual_size_choices_i[idx]
            if len(cat_actual_size_choice) == 0:
                continue
            sampled_actual_size = np.random.choice(cat_actual_size_choice)
            actual_size_choices_i[idx].remove(sampled_actual_size)
            selected_dict = {
                'synsetId' : row['synsetId'],
                'catId' : row['catId'],
                'ShapeNetModelId' : row['ShapeNetModelId'],
                'objId' : row['objId'],
                'size' : sampled_actual_size,
                'scale' : [sampled_actual_size / 0.9] * 3,
                'half_or_whole' : row['half_or_whole'],
                'perch_rot_angle' : row['perch_rot_angle'],
                'position' : np.asarray([x,y]),
            }
            selected_objects_i.append(selected_dict)
            category_list.append(row['catId'])
            object_list.append(row['objId'])

        selected_objects.append(selected_objects_i)
    return selected_objects

def generate_random(args):
    selected_objects = get_selected_objects(args)
    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        create_one_6d_scene(acc_scene_num, selected_objects[scene_num], args)

def generate_blender(args):
    selected_objects = get_selected_objects(args)
    all_yaml_file_name = []
    scene_num_to_selected = {}
    for scene_num in range(args.num_scenes):
        acc_scene_num = scene_num + args.start_scene_idx
        scene_folder_path = None
        try:
            scene_num_to_selected[acc_scene_num] = selected_objects[scene_num]
            blender_proc_scene = BlenderProcScene(acc_scene_num, selected_objects[scene_num], args)
            scene_folder_path = blender_proc_scene.scene_folder_path
            blender_proc_scene.output_yaml()
            all_yaml_file_name.append(blender_proc_scene.yaml_fname)
        except:
            print('##################################### GEN Error!')
            if scene_folder_path is not None:
                shutil.rmtree(scene_folder_path)
            traceback.print_exc()

    # output to .sh file
    output_save_dir = os.path.join(args.scene_save_dir, args.train_or_test)
    sh_fname = os.path.join(output_save_dir, 'blender_proc.sh')
    output_fid = open(sh_fname, 'w+', encoding='utf-8')
    tmp_dir = os.path.join(args.scene_save_dir, 'tmp')
    for yaml_fname in all_yaml_file_name:
        line = f'python run.py {yaml_fname} --temp-dir {tmp_dir}\n'
        output_fid.write(line)
    output_fid.close()

    selected_object_file = os.path.join(
        output_save_dir, 
        'selected_objects_{}_{}.pkl'.format(args.start_scene_idx, args.num_scenes),
    )
    with open(selected_object_file, 'wb+') as fh:
        pickle.dump(scene_num_to_selected, fh)

if __name__ == '__main__':
    if args.from_file:
        run_all_images(args)
    elif args.blender_proc:
        generate_blender(args)
    else:
        generate_random(args)
    