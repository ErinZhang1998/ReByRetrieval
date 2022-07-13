


## Running with PERCH2.0

```
source /opt/ros/kinetic/setup.bash
cd ros_python3_ws/
catkin init
catkin build object_recognition_node

roscore& #skip this if roscore is running outside
source /ros_python3_ws/devel/setup.bash 
Xvfb :5 -screen 0 800x600x24 & export DISPLAY=:5; 
cd /ros_python3_ws/src/perception/sbpl_perception/src/scripts/tools/fat_dataset
```

```
CUDA_VISIBLE_DEVICES=0 python fat_pose_image_custom.py --ros_node_name_suffix 00 --config config_custom_docker_real_
world.yaml --model_dir /data/custom_dataset/real_world_preselect_dv4_models --perch_debug_dir /data/custom_dataset/p
erch_output/real_world-genial-energy-25-preselect_dv4-14-oct30 --python_debug_dir /data/custom_dataset/model_output/
real_world-genial-energy-25-preselect_dv4-14-oct30 --annotation_list_path /data/custom_dataset/perch_pickles/2021-10
-30_real_world-genial-energy-25-preselect_dv4-14-oct30_01.pickle
```