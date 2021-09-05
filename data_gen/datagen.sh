! CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_file configs/blender_proc/config.yaml \
--init_method tcp://localhost:2010 \
--only_test \
--feature_extract \
--calculate_triplet_loss False \
--only_test_epoch 20 \
--model_path /raid/xiaoyuz1/blender_proc/result/copper-butterfly-13/models/20.pth \
--experiment_save_dir /raid/xiaoyuz1/blender_proc/result/copper-butterfly-13/preselect_dv4 \
--testing_scene_dir /raid/xiaoyuz1/blender_proc/preselect_dv4 \
--testing_yaml_file_dir /raid/xiaoyuz1/blender_proc/preselect_dv4/yaml_files \
--csv_file_path /raid/xiaoyuz1/preselect_all/preselect_table_top.csv
        
! CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_file configs/blender_proc/config.yaml \
--init_method tcp://localhost:2010 \
--only_test \
--feature_extract \
--calculate_triplet_loss False \
--only_test_epoch 20 \
--model_path /raid/xiaoyuz1/blender_proc/result/copper-butterfly-13/models/20.pth \
--experiment_save_dir /raid/xiaoyuz1/blender_proc/result/copper-butterfly-13/testing_set_dv \
--testing_scene_dir /raid/xiaoyuz1/blender_proc/testing_set_dv \
--testing_yaml_file_dir /raid/xiaoyuz1/blender_proc/testing_set_dv/yaml_files \
--csv_file_path /raid/xiaoyuz1/preselect_all/preselect_table_top.csv

! PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
--num_lights 1 \
--upright_ratio 1 \
--wall_unit_x 0.3 \
--wall_unit_y 0.3 \
--table_size 1 \
--table_size_xyz 1.5 1.5 1 \
--top_dir /raid/xiaoyuz1/blender_proc \
--shapenet_convex_decomp_dir /raid/xiaoyuz1/convex_decomposed \
--blender_proc \
--blender_model_root_dir /raid/xiaoyuz1/blender_proc/models \
--cctextures_path /raid/xiaoyuz1/cctextures \
--texture_to_use_file /raid/xiaoyuz1/cctextures_to_use.pkl \
--blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
--scene_save_dir /raid/xiaoyuz1/blender_proc \
--train_or_test preselect_dv3 \
--shapenet_filepath /raid/xiaoyuz1/ShapeNetCore.v2 \
--csv_file_path /raid/xiaoyuz1/preselect_dv3/preselect_table_top.csv \
--width 640 \
--height 480 \
--start_scene_idx 0 \
--num_scenes  600 \
--min_num_objects 15 \
--max_num_objects 15

! PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
--num_lights 1 \
--upright_ratio 1 \
--wall_unit_x 0.3 \
--wall_unit_y 0.3 \
--table_size 1 \
--table_size_xyz 1.5 1.5 1 \
--top_dir /raid/xiaoyuz1/blender_proc \
--shapenet_convex_decomp_dir /raid/xiaoyuz1/convex_decomposed \
--blender_proc \
--blender_model_root_dir /raid/xiaoyuz1/blender_proc/models \
--cctextures_path /raid/xiaoyuz1/cctextures \
--texture_to_use_file /raid/xiaoyuz1/cctextures_to_use.pkl \
--blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
--scene_save_dir /raid/xiaoyuz1/blender_proc \
--train_or_test preselect_dv4 \
--shapenet_filepath /raid/xiaoyuz1/ShapeNetCore.v2 \
--csv_file_path /raid/xiaoyuz1/preselect_dv4/preselect_table_top.csv \
--width 640 \
--height 480 \
--start_scene_idx 0 \
--num_scenes  800 \
--min_num_objects 15 \
--max_num_objects 15

PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
--num_lights 1 \
--upright_ratio 1 \
--wall_unit_x 0.3 \
--wall_unit_y 0.3 \
--table_size 1 \
--table_size_xyz 1.5 1.5 1 \
--top_dir /media/hdd/xiaoyuz1/blender_proc \
--shapenet_convex_decomp_dir /media/hdd/xiaoyuz1/convex_decomposed \
--blender_proc \
--blender_model_root_dir /media/hdd/xiaoyuz1/blender_proc/models \
--cctextures_path /media/hdd/xiaoyuz1/cctextures \
--texture_to_use_file /media/hdd/xiaoyuz1/cctextures_to_use.pkl \
--blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
--scene_save_dir /media/hdd/xiaoyuz1/blender_proc \
--train_or_test testing_set_dv \
--shapenet_filepath /media/hdd/xiaoyuz1/ShapeNetCore.v2 \
--csv_file_path /media/hdd/xiaoyuz1/preselect_test/preselect_table_top.csv \
--width 640 \
--height 480 \
--start_scene_idx 0 \
--num_scenes  150 \
--min_num_objects 15 \
--max_num_objects 15


PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
--num_lights 1 \
--upright_ratio 1 \
--wall_unit_x 0.3 \
--wall_unit_y 0.3 \
--table_size 1 \
--table_size_xyz 1.5 1.5 1 \
--top_dir /media/hdd/xiaoyuz1/blender_proc \
--shapenet_convex_decomp_dir /media/hdd/xiaoyuz1/convex_decomposed \
--blender_proc \
--blender_model_root_dir /media/hdd/xiaoyuz1/blender_proc/models \
--cctextures_path /media/hdd/xiaoyuz1/cctextures \
--texture_to_use_file /media/hdd/xiaoyuz1/cctextures_to_use.pkl \
--blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
--scene_save_dir /media/hdd/xiaoyuz1/blender_proc \
--train_or_test training_set_dv2 \
--shapenet_filepath /media/hdd/xiaoyuz1/ShapeNetCore.v2 \
--csv_file_path /media/hdd/xiaoyuz1/preselect_dv2/preselect_table_top.csv \
--width 640 \
--height 480 \
--start_scene_idx 0 \
--num_scenes  400 \
--min_num_objects 15 \
--max_num_objects 15

PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
--num_lights 1 \
--upright_ratio 1 \
--wall_unit_x 0.3 \
--wall_unit_y 0.3 \
--table_size 1 \
--table_size_xyz 1.5 1.5 1 \
--top_dir /media/hdd/xiaoyuz1/blender_proc \
--shapenet_convex_decomp_dir /media/hdd/xiaoyuz1/convex_decomposed \
--blender_proc \
--blender_model_root_dir /media/hdd/xiaoyuz1/blender_proc/models \
--cctextures_path /media/hdd/xiaoyuz1/cctextures \
--texture_to_use_file /media/hdd/xiaoyuz1/cctextures_to_use.pkl \
--blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
--scene_save_dir /media/hdd/xiaoyuz1/blender_proc \
--train_or_test training_set_dv1 \
--shapenet_filepath /media/hdd/xiaoyuz1/ShapeNetCore.v2 \
--csv_file_path /media/hdd/xiaoyuz1/preselect_dv1/preselect_table_top.csv \
--width 640 \
--height 480 \
--start_scene_idx 0 \
--num_scenes 200 \
--min_num_objects 15 \
--max_num_objects 15

# PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
# --num_lights 1 \
# --upright_ratio 1 \
# --wall_unit_x 0.3 \
# --wall_unit_y 0.3 \
# --table_size 1 \
# --table_size_xyz 1.5 1.5 1 \
# --top_dir /media/hdd/xiaoyuz1/blender_proc \
# --shapenet_convex_decomp_dir /media/hdd/xiaoyuz1/convex_decomposed \
# --blender_proc \
# --blender_model_root_dir /media/hdd/xiaoyuz1/blender_proc/models \
# --cctextures_path /media/hdd/xiaoyuz1/cctextures \
# --texture_to_use_file /media/hdd/xiaoyuz1/cctextures_to_use.pkl \
# --blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
# --scene_save_dir /media/hdd/xiaoyuz1/blender_proc \
# --train_or_test testing_set_new \
# --shapenet_filepath /media/hdd/xiaoyuz1/ShapeNetCore.v2 \
# --csv_file_path /media/hdd/xiaoyuz1/preselect_august/preselect_table_top_test.csv \
# --width 640 \
# --height 480 \
# --start_scene_idx 0 \
# --num_scenes 100 \
# --min_num_objects 15 \
# --max_num_objects 15


# PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
# --blender_proc \
# --blender_model_root_dir /media/hdd/xiaoyuz1/blender_proc/models \
# --blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
# --scene_save_dir /media/hdd/xiaoyuz1/blender_proc \
# --train_or_test training_set_new \
# --shapenet_filepath /media/hdd/xiaoyuz1/ShapeNetCore.v2 \
# --shapenet_convex_decomp_dir /media/hdd/xiaoyuz1/convex_decomposed \
# --top_dir /media/hdd/xiaoyuz1/blender_proc \
# --csv_file_path /media/hdd/xiaoyuz1/preselect_august/preselect_table_top_train.csv \
# --width 640 \
# --height 480 \
# --start_scene_idx 0 \
# --num_scenes 500 \
# --num_lights 1 \
# --min_num_objects 15 \
# --max_num_objects 15 \
# --upright_ratio 1 \
# --wall_unit_x 0.3 \
# --wall_unit_y 0.3 \
# --table_size 1 \
# --table_size_xyz 1.5 1.5 1 

##########################################################################################

# PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
# --blender_proc \
# --blender_model_root_dir /raid/xiaoyuz1/blender_proc/models \
# --blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
# --scene_save_dir /raid/xiaoyuz1/blender_proc \
# --train_or_test testing_set_2 \
# --shapenet_filepath /raid/xiaoyuz1/ShapeNetCore.v2 \
# --shapenet_convex_decomp_dir /raid/xiaoyuz1/convex_decomposed \
# --top_dir /raid/xiaoyuz1/perch \
# --csv_file_path /raid/xiaoyuz1/new_august_21/preselect_table_top.csv \
# --width 640 \
# --height 480 \
# --start_scene_idx 0 \
# --num_scenes 100 \
# --num_lights 1 \
# --min_num_objects 10 \
# --max_num_objects 10 \
# --upright_ratio 1 \
# --wall_unit_x 0.3 \
# --wall_unit_y 0.3 \
# --table_size 1 \
# --table_size_xyz 1.5 1.5 1 

# PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
# --single_object \
# --blender_proc \
# --blender_model_root_dir /raid/xiaoyuz1/blender_proc/models \
# --blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
# --scene_save_dir /raid/xiaoyuz1/blender_proc \
# --train_or_test inference_set \
# --shapenet_filepath /raid/xiaoyuz1/ShapeNetCore.v2 \
# --shapenet_convex_decomp_dir /raid/xiaoyuz1/convex_decomposed \
# --top_dir /raid/xiaoyuz1/perch \
# --csv_file_path /raid/xiaoyuz1/all_2021/preselect_table_top.csv \
# --width 640 \
# --height 480 \
# --start_scene_idx 0 \
# --num_scenes 3000 \
# --num_lights 1 \
# --min_num_objects 20 \
# --max_num_objects 20 \
# --upright_ratio 1 \
# --wall_unit_x 0.3 \
# --wall_unit_y 0.3 \
# --table_size 1 \
# --table_size_xyz 1.5 1.5 1 



# PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
# --blender_proc \
# --blender_model_root_dir /raid/xiaoyuz1/blender_proc/models \
# --blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
# --scene_save_dir /raid/xiaoyuz1/blender_proc \
# --train_or_test testing_set \
# --shapenet_filepath /raid/xiaoyuz1/ShapeNetCore.v2 \
# --shapenet_convex_decomp_dir /raid/xiaoyuz1/convex_decomposed \
# --top_dir /raid/xiaoyuz1/perch \
# --csv_file_path /raid/xiaoyuz1/preselect_july_perch_split_2021/preselect_table_top_test.csv \
# --width 640 \
# --height 480 \
# --start_scene_idx 100 \
# --num_scenes 100 \
# --num_lights 1 \
# --min_num_objects 20 \
# --max_num_objects 20 \
# --upright_ratio 1 \
# --wall_unit_x 0.3 \
# --wall_unit_y 0.3 \
# --table_size 1 \
# --table_size_xyz 1.5 1.5 1 


# PYTHONPATH=/home/xiaoyuz1/retrieve python data_gen/datagen_blender_proc.py \
# --blender_proc \
# --blender_model_root_dir /raid/xiaoyuz1/blender_proc/models \
# --blender_proc_config /home/xiaoyuz1/retrieve/data_gen/blender_proc_self.yaml \
# --scene_save_dir /raid/xiaoyuz1/blender_proc \
# --train_or_test training_set_2 \
# --shapenet_filepath /raid/xiaoyuz1/ShapeNetCore.v2 \
# --shapenet_convex_decomp_dir /raid/xiaoyuz1/convex_decomposed \
# --top_dir /raid/xiaoyuz1/perch \
# --csv_file_path /raid/xiaoyuz1/preselect_july_perch_split_2021/preselect_table_top_train.csv \
# --width 640 \
# --height 480 \
# --start_scene_idx 250 \
# --num_scenes 150 \
# --num_lights 1 \
# --min_num_objects 20 \
# --max_num_objects 20 \
# --upright_ratio 1 \
# --wall_unit_x 0.3 \
# --wall_unit_y 0.3 \
# --table_size 1 \
# --table_size_xyz 1.5 1.5 1 