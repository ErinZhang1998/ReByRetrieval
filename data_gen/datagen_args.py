import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--shapenet_filepath", dest="shapenet_filepath", default='/raid/xiaoyuz1/ShapeNetCore.v2')
parser.add_argument("--shapenet_convex_decomp_dir", dest="shapenet_convex_decomp_dir")
parser.add_argument("--top_dir", dest="top_dir", help='Directory where base.xml resides')
parser.add_argument("--csv_file_path", dest="csv_file_path", help='CSV file of Shapenet object model annotations')
parser.add_argument("--train_or_test", dest="train_or_test", default='training_set')
parser.add_argument("--scene_save_dir", dest="scene_save_dir")
parser.add_argument("--num_scenes", dest="num_scenes", type=int, default=1)
parser.add_argument("--min_num_objects", dest="min_num_objects", type=int, default=3)
parser.add_argument("--max_num_objects", dest="max_num_objects", type=int, default=6)
parser.add_argument("--start_scene_idx", dest="start_scene_idx", type=int, default=0)
parser.add_argument("--num_lights", dest="num_lights", type=int, default=3)
parser.add_argument("--depth_factor", dest="depth_factor", type=int, default=1000)
parser.add_argument("--width", dest="width", type=int, default=640, help='Width of output images')
parser.add_argument("--height", dest="height", type=int, default=480, help='Height of output images')
parser.add_argument("--use_walls", dest="use_walls", type=int, default=1, help='NO USAGE')
parser.add_argument("--table_size", dest="table_size", type=float, default=2)
parser.add_argument("--debug", action="store_true", dest="debug")
parser.add_argument("--upright_ratio", dest="upright_ratio", type=float, default=0.5)
parser.add_argument("--single_object", action="store_true", dest="single_object")
parser.add_argument("--wall_unit_x", dest="wall_unit_x", type=float, default=0.3)
parser.add_argument("--wall_unit_y", dest="wall_unit_y", type=float, default=0.3)

parser.add_argument("--not_step_mujoco", action="store_true", dest="not_step_mujoco")
parser.add_argument("--predefined_camera_locations", action="store_true", dest="predefined_camera_locations")
parser.add_argument("--from_file", action="store_true", dest="from_file")

parser.add_argument("--scene_dir", dest="scene_dir", help='scene directory')
parser.add_argument("--perch_dir", dest="perch_dir", help='directory that stores all perch_output')



parser.add_argument("--perch_annotation_file", dest="perch_annotation_file", help='output_poses.txt file from perch')
parser.add_argument("--json_annotation_file", dest="json_annotation_file", help='annotations_i.json file for one image to reconstruct scene')
parser.add_argument("--camera_annotation_file", dest="camera_annotation_file", help='annotations.json file for getting camera locations and targets from the `images` fileds')

# For blender proc
parser.add_argument("--blender_proc", action="store_true", dest="blender_proc")
parser.add_argument("--blender_model_root_dir", dest="blender_model_root_dir", default='/raid/xiaoyuz1/perch/blender_models')
# parser.add_argument("--output_root_dir", dest="output_root_dir", default='/raid/xiaoyuz1/perch/blender_proc')


args = parser.parse_args()