from optparse import OptionParser

parser = OptionParser()
parser.add_option("--shapenet_filepath", dest="shapenet_filepath")

# shapenet_convex_decomp_dir shapenet_decomp_filepath
parser.add_option("--shapenet_convex_decomp_dir", dest="shapenet_convex_decomp_dir")

parser.add_option("--top_dir", dest="top_dir")

parser.add_option("--csv_file_path", dest="csv_file_path")

parser.add_option("--train_or_test", dest="train_or_test", default='training_set')
parser.add_option("--scene_save_dir", dest="scene_save_dir")

parser.add_option("--num_scenes", dest="num_scenes", type="int", default=1)
parser.add_option("--min_num_objects", dest="min_num_objects", type="int", default=3)
parser.add_option("--max_num_objects", dest="max_num_objects", type="int", default=6)

parser.add_option("--start_scene_idx", dest="start_scene_idx", type="int", default=0)
parser.add_option("--num_lights", dest="num_lights", type="int", default=3)

parser.add_option("--depth_factor", dest="depth_factor", type="int", default=1000)
parser.add_option("--width", dest="width", type="int", default=640)
parser.add_option("--height", dest="height", type="int", default=480)
parser.add_option("--use_walls", dest="use_walls", type="int", default=1)
parser.add_option("--table_size", dest="table_size", type="int", default=2)
parser.add_option("--debug", action="store_true", dest="debug")
parser.add_option("--upright_ratio", dest="upright_ratio", type="float", default=0.5)

parser.add_option("--single_object", action="store_true", dest="single_object")

# parser.add_option("--camera_radius", dest="camera_radius", type="float", default=0)

(args, argss) = parser.parse_args()
