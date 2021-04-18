import trimesh
from optparse import OptionParser
import os

parser = OptionParser()
parser.add_option("--shapenet_filepath", dest="shapenet_filepath")


parser.add_option("--shapenet_obj_id", dest="shapenet_obj_id")
parser.add_option("--shapenet_obj_cat", dest="shapenet_obj_cat")
(args, argss) = parser.parse_args()

obj_mesh_filename = os.path.join(args.shapenet_filepath,'0{}/{}/models/model_normalized.obj'.format(args.shapenet_obj_cat, args.shapenet_obj_id))
object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
object_mesh.show()