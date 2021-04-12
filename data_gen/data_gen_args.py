from optparse import OptionParser

parser = OptionParser()
parser.add_option("--shapenet_filepath", dest="shapenet_filepath")

parser.add_option("--shapenet_decomp_filepath", dest="shapenet_decomp_filepath")

parser.add_option("--top_dir", dest="top_dir")

parser.add_option("--csv_file_path", dest="csv_file_path")

parser.add_option("--train_or_test", dest="train_or_test", default='training_set')

parser.add_option("--num_scenes", dest="num_scenes", type="int", default=3)
parser.add_option("--min_num_objects", dest="min_num_objects", type="int", default=3)
parser.add_option("--max_num_objects", dest="max_num_objects", type="int", default=6)

parser.add_option("--start_scene_idx", dest="start_scene_idx", type="int", default=0)
(args, argss) = parser.parse_args()
