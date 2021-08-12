import traceback
import numpy as np
import pickle
import pandas as pd

from datagen_args import *
from utils.datagen_utils import *
from perch_scene import *


def run_single_object(acc_scene_num, selected_objects, args):
    scene_folder_path = None
    try:
        perch_scene = PerchScene(acc_scene_num, selected_objects[acc_scene_num], args)
        scene_folder_path = perch_scene.scene_folder_path
        perch_scene.create_convex_decomposed_scene()
        perch_scene.create_camera_scene()
        return scene_folder_path
    except:
        print('##################################### GEN Error!', scene_folder_path)
        if scene_folder_path is not None:
            shutil.rmtree(scene_folder_path)
        print(selected_objects)
        traceback.print_exc()
        return scene_folder_path

if __name__ == '__main__':
    # np.random.seed(129)
    df = pd.read_csv(args.csv_file_path)

    scale_choices = {}
    for i in range(len(df)):
        scale_choices[i] = [1.0] #[0.75, 0.85, 1.0]
    
    selected_objects = []
    for i in range(len(df)):
        row = df.iloc[i]
        for sample_scale in scale_choices[i]:
            selected_objects_i = [(
                row['synsetId'], 
                row['catId'], 
                row['ShapeNetModelId'], 
                row['objId'], 
                sample_scale,
                row['half_or_whole'],
                row['perch_rot_angle']
            )]
            selected_objects.append(selected_objects_i)
    
    # only_run = [246, 297, 324]
    print("*** TOTAL: ", len(selected_objects))
    problematic_generation = []
    for acc_scene_num in range(len(selected_objects)):
        # if acc_scene_num not in only_run:
        #     continue
        scene_folder_path = None
        try:
            perch_scene = PerchScene(acc_scene_num, selected_objects[acc_scene_num], args)
            scene_folder_path = perch_scene.scene_folder_path
            perch_scene.create_convex_decomposed_scene()
            perch_scene.create_camera_scene()
        except:
            print('##################################### GEN Error!', scene_folder_path)
            if scene_folder_path is not None:
                shutil.rmtree(scene_folder_path)
            print(selected_objects)
            traceback.print_exc()
            problematic_generation.append(selected_objects[acc_scene_num])
    
    with open(os.path.join(args.top_dir, 'problem.pkl'), 'wb+') as fp:
        pickle.dump(problematic_generation, fp)


    
