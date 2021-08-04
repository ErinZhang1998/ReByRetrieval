from optparse import OptionParser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shapenet_filepath", dest="shapenet_filepath")

# shapenet_convex_decomp_dir shapenet_decomp_filepath
parser.add_argument("--shapenet_convex_decomp_dir", dest="shapenet_convex_decomp_dir")

parser.add_argument("--top_dir", dest="top_dir")

parser.add_argument("--csv_file_path", dest="csv_file_path")

parser.add_argument("--train_or_test", dest="train_or_test", default='training_set')
parser.add_argument("--scene_save_dir", dest="scene_save_dir")

parser.add_argument("--num_scenes", dest="num_scenes", type=int, default=1)
parser.add_argument("--min_num_objects", dest="min_num_objects", type=int, default=3)
parser.add_argument("--max_num_objects", dest="max_num_objects", type=int, default=6)

parser.add_argument("--start_scene_idx", dest="start_scene_idx", type=int, default=0)
parser.add_argument("--num_lights", dest="num_lights", type=int, default=3)

parser.add_argument("--depth_factor", dest="depth_factor", type=int, default=1000)
parser.add_argument("--width", dest="width", type=int, default=640)
parser.add_argument("--height", dest="height", type=int, default=480)
parser.add_argument("--use_walls", dest="use_walls", type=int, default=1)
parser.add_argument("--table_size", dest="table_size", type=int, default=2)
parser.add_argument("--debug", action="store_true", dest="debug")
parser.add_argument("--upright_ratio", dest="upright_ratio", type=float, default=0.5)
parser.add_argument("--single_object", action="store_true", dest="single_object")

# parser.add_argument("--canonical_size", dest="canonical_size", nargs="+", default=[0.3, 0.3, 0.3])
parser.add_argument("--canonical_size", dest="canonical_size", type=float, default=0.3)
parser.add_argument("--wall_unit_x", dest="wall_unit_x", type=float, default=0.3)
parser.add_argument("--wall_unit_y", dest="wall_unit_y", type=float, default=0.3)

parser.add_argument("--size_xyz", action="store_true", dest="size_xyz")

# parser.add_argument("--camera_radius", dest="camera_radius", type=float, default=0)
# (args, argss) = parser.parse_args()
args = parser.parse_args()