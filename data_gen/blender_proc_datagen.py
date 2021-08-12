import os
import json 
import yaml
import shutil
import argparse
import numpy as np 
from scipy.spatial.transform import Rotation as R, rotation        

import trimesh

import utils.blender_proc_utils as bp_utils
import utils.datagen_utils as datagen_utils
# camera_pos_dir = os.path.join(options.output_root_dir, 'camera_positions')
# yaml_dir = os.path.join(options.output_root_dir, 'yaml_files')
# if not os.path.exists(yaml_dir):
#     os.mkdir(yaml_dir)
# if not os.path.exists(camera_pos_dir):
#     os.mkdir(camera_pos_dir)

# json_file_name = '/raid/xiaoyuz1/perch/perch_balance/testing_set_3/scene_000010/annotations.json'
# coco_anno = json.load(open(json_file_name))

# experiment_name = '_'.join(json_file_name.split('/')[-3:-1])
# print("Experiment name: ", experiment_name)
# blender_proc_output_dir = os.path.join(options.output_root_dir, experiment_name)
# print("Output directory name: ", blender_proc_output_dir)

class BlenderProcScene(object):
    
    def __init__(self, scene_num, selected_objects, args):
        self.args = args
        self.scene_num = scene_num
        self.selected_objects = selected_objects
        self.shapenet_filepath, top_dir, self.train_or_test = args.shapenet_filepath, args.top_dir, args.train_or_test
        self.num_lights = args.num_lights
        self.num_objects = len(selected_objects) 
        
        self.width = args.width
        self.height = args.height 
        
        self.scene_save_dir = args.scene_save_dir

        if not os.path.exists(args.scene_save_dir):
            os.mkdir(args.scene_save_dir)
        
        output_save_dir = os.path.join(args.scene_save_dir, self.train_or_test)
        if not os.path.exists(output_save_dir):
            os.mkdir(output_save_dir)
        self.output_save_dir = output_save_dir
        
        self.scene_name = f'{self.train_or_test}_scene_{scene_num:06}' 
        scene_folder_path = os.path.join(output_save_dir, f'scene_{scene_num:06}')
        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.mkdir(scene_folder_path)
        self.scene_folder_path = scene_folder_path

        yaml_dir = os.path.join(args.scene_save_dir, 'yaml_files')
        if not os.path.exists(yaml_dir):
            os.mkdir(yaml_dir)
        self.yaml_dir = yaml_dir

        convex_decomposition_cache_path = os.path.join(self.args.scene_save_dir, "convex_decomposition_cache")
        if not os.path.exists(convex_decomposition_cache_path):
            os.mkdir(convex_decomposition_cache_path)
        # convex_decomposition_cache_path = os.path.join(convex_decomposition_cache_path, self.scene_name)
        # if not os.path.exists(convex_decomposition_cache_path):
        #     os.mkdir(convex_decomposition_cache_path)
        self.convex_decomposition_cache_path = convex_decomposition_cache_path

        # self.create_table()
        self.object_info_dict = dict()
        # for object_idx in range(self.num_objects):
        #     self.create_object(object_idx)
        # self.total_camera_num = 0
        # self.camera_info_dict = dict()

    def add_camera_to_scene(self):
        # "cp_is_object": True,
        # float(self.table_info.height + 0.1)
        location_dict = {
            "provider": "sampler.UpperRegionSampler",
            "to_sample_on": {
                "provider": "getter.Entity",
                "index": 0,
                "conditions": {
                    "cp_shape_net_table": True,
                    "type": "MESH",
                }
            },
            "min_height": 0.5,
            "max_height": 1,
            "use_ray_trace_check": False,
        }
        # location_dict = {
        #     "provider": "sampler.Shell",
        #     "center": {
        #         "provider": "getter.POI",
        #         "selector": {
        #             "provider": "getter.Entity",
        #             "conditions": {
        #                 "cp_is_object": True,
        #                 "type": "MESH"
        #             }
        #         }
        #     },
        #     "radius_min": 0.2,
        #     "radius_max": 0.5,
        #     "elevation_min": 30,
        #     "elevation_max": 60,
        #     "uniform_elevation": True
        # }

        rotation_dict = {
            "format": "look_at",
            "value": {
                "provider": "getter.AttributeMerger",
                "elements": [
                    {
                        "provider": "getter.POI",
                        "selector": {
                            "provider": "getter.Entity",
                            "conditions": {
                                "cp_is_object": True,
                                "type": "MESH"
                            }
                        }
                    },
                    {
                        "provider": "sampler.Uniform3d",
                        "min": [-0.1]*3,
                        "max": [0.1]*3,
                    },
                ],
                "transform_by": "sum"
            },
        }
        rotation_dict = {
            "format": "look_at",
            "value": {
                "provider": "getter.POI",
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_is_object": True,
                        "type": "MESH"
                    }
                }
            },
        }
        '''
        "inplane_rot": {
                "provider": "sampler.Value",
                "type": "float",
                "min": -0.7854,
                "max": 0.7854
            }
        '''

        # rotation_dict = {
        #     "format": "look_at",
        #     "value": {
        #         "provider": "getter.POI"
        #     },
        #     "inplane_rot": {
        #         "provider": "sampler.Value",
        #         "type": "float",
        #         "min": -0.7854,
        #         "max": 0.7854
        #     }
        # }

        camera_pose_sampler = {
            "number_of_samples": 5,
            "check_if_objects_visible": {
                "provider": "getter.Entity",
                "conditions": {
                    "cp_is_object": True,
                    "type": "MESH"
                }
            },
            "location" : location_dict,
            "rotation" : rotation_dict,
        }

        camera_pose_sampler = {
            "number_of_samples": 5,
            "check_if_objects_visible": {
                "provider": "getter.Entity",
                "conditions": {
                    "cp_physics": True,
                    "type": "MESH"
                }
            },
            "location": {
                "provider":"sampler.Uniform3d",
                "max":[1, 1, float(0.5+self.table_info.height)],
                "min":[-1, -1, float(0.2+self.table_info.height)]
            },
            "rotation": {
                "format": "look_at",
                "value": {
                    "provider": "getter.POI",
                    "selector": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "cp_physics": True,
                            "type": "MESH"
                        }
                    }
                },
            }
        }
        
        camera_sampler_dict = {
            "module": "camera.CameraSampler",
            "config" : {
                "cam_poses": [ 
                    camera_pose_sampler,
                ],
                "intrinsics": {
                    "cam_K": [376.72453850819767, 0.0, 320.0, 0.0, 376.72453850819767, 240.0, 0.0, 0.0, 1.0],
                    "resolution_x": self.width,
                    "resolution_y": self.height,
                    "fov": 45,
                }
            }
        }

        camera_sampler_dict = {
            "module": "camera.CameraSampler",
            "config": {
            "cam_poses": [
            {
                "number_of_samples": 5,
                "location": {
                "provider":"sampler.PartSphere",
                "center": {
                    "provider": "getter.POI",
                    "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_category_id": 0,
                        "type": "MESH"
                    }
                    }
                },
                "distance_above_center": 0.5,
                "radius": 1,
                "mode": "SURFACE"
                },
                "rotation": {
                "format": "look_at",
                "value": {
                    "provider": "getter.POI",
                    "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_category_id": 0,
                        "type": "MESH"
                    }
                    }
                }
                }
            }
            ]
            }
        }

        json_file_name = '/raid/xiaoyuz1/perch/perch_balance/testing_set_3/scene_000010/annotations.json'
        coco_anno = json.load(open(json_file_name))

        camera_pose_fname = os.path.join(self.scene_folder_path, '{}_CameraPositions'.format(self.scene_name))
        camear_pose_output_fid = open(camera_pose_fname, 'w+', encoding='utf-8')
        # cam_poses = []
        for ann in coco_anno['images']:
            euler_rot = R.from_quat(ann['rot_quat']).as_euler('xyz')
            euler_rot = [float(item) for item in euler_rot]
            location = [float(item) for item in ann['pos']]
            # pose_dict = {
            #     "location" : ann['pos'],
            #     "rotation" : {
            #         "value" : euler_rot,
            #     }
            # }
            # cam_poses.append(pose_dict)

            line = f'{location[0]} {location[1]} {location[2]} {euler_rot[0]} {euler_rot[1]} {euler_rot[2]}\n'
            camear_pose_output_fid.write(line)
        camear_pose_output_fid.close()

        camera_sampler_dict = {
            "module": "camera.CameraLoader",
            "config": {
            "path": camera_pose_fname,
            "file_format": "location rotation/value",
            "intrinsics": {
                "cam_K": [376.72453850819767, 0.0, 320.0, 0.0, 376.72453850819767, 240.0, 0.0, 0.0, 1.0],
                "resolution_x": 640,
                "resolution_y": 480,
            }
            }
        }


        return [camera_sampler_dict]


    def add_lights_to_scene(self):
        light_module = [
            {
            "module": "lighting.LightLoader",
            "config": {
                "lights": [
                {
                    "type": "POINT",
                    "location": [1, 1, 1],
                    "energy": 1000
                }
                ]
            }
            }
        ]
        light_location = {
            "provider": "sampler.Shell",
            "center": {
                "provider": "getter.POI",
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_is_object": True,
                        "type": "MESH",
                    }
                }
            },
            "radius_min": 1,
            "radius_max": 5,
            "elevation_min": 1,
            "elevation_max": 89,
            "uniform_elevation": True
        }
        light_module = [{
            "module": "lighting.LightSampler",
            "config": {
                "lights": [
                    {
                        "location": light_location,
                        "type": "POINT",
                        "energy": {
                            "provider": "sampler.Value",
                            "type": "int",
                            "min": 100,
                            "max": 1000
                        }
                    }
                ] * self.num_lights 
            }
        }]
        return light_module
    
    def save_object_to_file(self, ann):
        shapenet_dir = os.path.join(
            self.shapenet_filepath,
            '{}/{}'.format(ann['synset_id'], ann['model_id']),
        )
        input_obj_file = os.path.join(shapenet_dir, 'models', 'model_normalized.obj')
        mtl_path = os.path.join(shapenet_dir, 'models', 'model_normalized.mtl')
        image_material_dir = os.path.join(shapenet_dir, 'images')

        model_dir = os.path.join(self.args.blender_model_root_dir, ann['model_name'])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_dir_models = os.path.join(model_dir, 'models')
        if not os.path.exists(model_dir_models):
            os.mkdir(model_dir_models)
        
        output_obj_file = os.path.join(model_dir_models, 'textured.obj')
        new_mtl_path = os.path.join(model_dir_models, 'model_normalized.mtl')
        shutil.copyfile(mtl_path, new_mtl_path)

        if os.path.exists(image_material_dir):
            new_image_dir = os.path.join(model_dir, 'images')
            # if not os.path.exists(new_image_dir):
            #     os.mkdir(new_image_dir)
            if os.path.exists(new_image_dir):
                shutil.rmtree(new_image_dir)
            
            shutil.copytree(image_material_dir, new_image_dir) 

        x,y,z = ann['actual_size']
        bp_utils.scale_obj_file(input_obj_file, output_obj_file, np.array([x,z,y]), add_name=ann['model_name'])

        # object_mesh = trimesh.load(output_obj_file, force='mesh')
        # 
        return output_obj_file
    
    # name, synset_id, model_id
    # actual_size, position, euler
    def create_table(self):
        synset_id = '04379243'
        table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
        table_mesh_fname = os.path.join(self.shapenet_filepath, f'{synset_id}/{table_id}/models/model_normalized.obj')
        self.table_info = bp_utils.BlenderProcTable(
            model_name=f'{self.train_or_test}_scene_{self.scene_num}_table',
            synset_id=synset_id,
            model_id=table_id,
            shapenet_file_name=table_mesh_fname,
            num_objects_in_scene=self.num_objects,
            table_size=self.args.table_size,
        )
        ann = self.table_info.get_blender_proc_dict()
        table_path = self.save_object_to_file(ann)
        table_module = {
            "module": "loader.ObjectLoader",
            "config": {
                "paths": [table_path],
                "add_properties": {
                    "cp_physics": False,
                    "cp_category_id" : self.num_objects,
                    "cp_shape_net_table" : True,
                }
            }
        }
        entity_manipulator = {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector" : {
                    "provider" : "getter.Entity",
                    "check_empty": True,
                    "conditions": {
                        "name": ann['model_name'],
                        "type": "MESH"  # this guarantees that the object is a mesh, and not for example a camera
                    }
                },
                "location" : [float(item) for item in ann['position']],
                "rotation_euler" : [float(item) for item in ann['euler']],
            },
        }
        return table_module, entity_manipulator

    def create_object(self, object_idx):
        synset_id = '0{}'.format(self.selected_objects[object_idx]['synsetId'])
        model_id = self.selected_objects[object_idx]['ShapeNetModelId']
        model_name=f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}'
        actual_size = self.selected_objects[object_idx]['size']
        
        shapnet_file_name = os.path.join(self.shapenet_filepath, f'{synset_id}/{model_id}/models/model_normalized.obj')
        object_info = bp_utils.BlenderProcNonTable(
            model_name=model_name,
            synset_id=synset_id,
            model_id=model_id,
            shapenet_file_name=shapnet_file_name,
            num_objects_in_scene=self.num_objects,
            table_size=self.args.table_size,
            object_idx=object_idx,
            selected_object_info=self.selected_objects[object_idx],
            table_height=self.table_info.height,
            upright_ratio=self.args.upright_ratio,
        )
        self.object_info_dict[object_idx] = object_info
        
        ann = {
            'synset_id' : synset_id,
            'model_id' : model_id,
            'model_name' : model_name,
            'actual_size' : actual_size,
        }
        ann = object_info.get_blender_proc_dict()
        output_obj_file = self.save_object_to_file(ann)
        object_module = {
            "module": "loader.ObjectLoader",
            "config": {
                "path": output_obj_file,
                "add_properties": {
                    "cp_physics": True,
                    "cp_category_id" : object_idx,
                    "cp_is_object" : True,
                }
            }
        }
        # 
        surface_pose_sampler = {
            "module": "object.OnSurfaceSampler",
            "config": {
                "objects_to_sample": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "name": model_name
                    }
                },
                "surface": {
                    "provider": "getter.Entity",
                    "index": 0,
                    "conditions": {
                       "cp_shape_net_table": True,
                       "type": "MESH",
                    }
                },
                "pos_sampler": {
                    "provider": "sampler.UpperRegionSampler",
                    "to_sample_on": {
                        "provider": "getter.Entity",
                        "index": 0,
                        "conditions": {
                            "cp_shape_net_table": True,
                            "type": "MESH",
                        }
                    },
                    "min_height": 0,
                    "max_height": 0.5,
                    "use_ray_trace_check": False,
                },
                "min_distance": 0.1,
                "max_distance": 0.5,
                "rot_sampler": {
                    "provider": "sampler.Uniform3d",
                    "max": [0,0,0],
                    "min": [0,0,6.28],
                }
            }
        }

        entity_manipulator = {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector" : {
                    "provider" : "getter.Entity",
                    "check_empty": True,
                    "conditions": {
                        "name": ann['model_name'],
                        "type": "MESH"  # this guarantees that the object is a mesh, and not for example a camera
                    }
                },
                "location" : [float(item) for item in ann['position']],
                "rotation_euler" : [float(item) for item in ann['euler']],
            },
        }
        # import pdb; pdb.set_trace()
        return object_module, surface_pose_sampler, entity_manipulator
        
    def output_yaml(self):
        table_module, table_manipulator = self.create_table()
        object_loader_module = [table_module]
        surface_sampler_module = []
        for object_idx in range(len(self.selected_objects)):
            object_module, surface_pose_sampler, object_manipulator = self.create_object(object_idx)
            object_loader_module += [object_module]
            surface_sampler_module += [object_manipulator]

        camera_module = self.add_camera_to_scene()
        light_module = self.add_lights_to_scene()
        rgb_renderer_module = [
            {
                "module": "renderer.RgbRenderer",
                "config": {
                    "output_key": "colors",
                    "samples": 350,
                    "render_distance": True
                }
            },
        ]
        write_module = [
            {
                "module": "writer.ShapeNetWriter"
            },
            {
                "module": "writer.ObjectStateWriter"
            },
            {
                "module": "writer.CameraStateWriter",
                "config": {
                    "attributes_to_write": ["location", "rotation_euler", "fov_x", "fov_y", "shift_x", "shift_y", "cam_K", "cam2world_matrix"]
                }
            },
            {
                "module": "writer.Hdf5Writer",
                "config": {
                    "delete_temporary_files_afterwards" : False,
                    "postprocessing_modules": {
                        "distance": [
                        {"module": "postprocessing.TrimRedundantChannels"},
                        {"module": "postprocessing.Dist2Depth"}
                        ]
                    }
                }
            }
        ]

        initialize_module = [
            {
            "module": "main.Initializer",
            "config": {
                "global": {
                "output_dir": self.scene_folder_path,
                }
            }
            }
        ]
        physics_positioning_module = [
            {
                "module": "object.PhysicsPositioning",
                "config": {
                    "min_simulation_time": 4,
                    "max_simulation_time": 20,
                    "check_object_interval": 1,
                    "collision_shape": "CONVEX_DECOMPOSITION",
                    "convex_decomposition_cache_path" : self.convex_decomposition_cache_path,
                }
            }
        ]
        modules = initialize_module 
        modules += object_loader_module
        modules += [table_manipulator]
        modules += surface_sampler_module
        modules = modules + light_module 
        modules = modules + camera_module 
        modules = modules + physics_positioning_module
        modules = modules + rgb_renderer_module
        modules = modules + write_module
        
        final_yaml_dict = {
            "version": 3,
            "setup": {
                "blender_install_path": "/home/<env:USER>/blender/",
                "pip": [
                    "h5py",
                    "imageio"
                ]
            },
            "modules" : modules,
        }
        # Output to yaml file
        yaml_fname = os.path.join(self.yaml_dir, '{}.yaml'.format(self.scene_name))
        print("Output to: ", yaml_fname)
        with open(yaml_fname, 'w+') as outfile:
            yaml.dump(final_yaml_dict, outfile, default_flow_style=False)


# obj_paths = []
# entity_manipulators = []
# for ann in coco_anno['categories']:
#     model_name = ann['name']
#     shapenet_dir = os.path.join(
#         shapenet_filepath,
#         '{}/{}/models'.format(ann['synset_id'], ann['model_id']),
#     )
#     input_obj_file = os.path.join(shapenet_dir,'model_normalized.obj')
#     mtl_path = os.path.join(shapenet_dir, 'model_normalized.mtl')

#     model_dir = os.path.join(blender_model_root_dir, model_name)
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     output_obj_file = os.path.join(model_dir, 'textured.obj')
#     new_mtl_path = os.path.join(model_dir, 'model_normalized.mtl')
#     shutil.copyfile(mtl_path, new_mtl_path)
    
#     size_x, size_y, size_z = ann['actual_size']
#     bp_utils.scale_obj_file(input_obj_file, output_obj_file, [size_x, size_z, size_y], add_name=ann['name'])
#     obj_paths.append(output_obj_file)

#     entity_manipulator = bp_utils.get_new_entity_manipulator()
#     selector_dict = {
#         "provider" : "getter.Entity",
#         "check_empty": True,
#         "conditions": {
#             "name": ann['name'],
#             "type": "MESH"  # this guarantees that the object is a mesh, and not for example a camera
#         }
#     }
#     r = R.from_quat(ann['quat'])
#     rot_euler = r.as_euler('xyz')
#     rot_euler = [float(item) for item in rot_euler] #[(1/2)*np.pi, 0, 0]
#     config_dict = {
#         "selector" : selector_dict,
#         "location" : ann['position'],
#         "rotation_euler" : rot_euler,
#         "cp_physics": True
#     }
#     entity_manipulator["config"] = config_dict
#     entity_manipulators.append(entity_manipulator)

# # Export camera pose
# camera_pose_fname = os.path.join(camera_pos_dir, '{}_CameraPositions'.format(experiment_name))
# camear_pose_output_fid = open(camera_pose_fname, 'w+', encoding='utf-8')
# # cam_poses = []
# for ann in coco_anno['images']:
#     euler_rot = R.from_quat(ann['rot_quat']).as_euler('xyz')
#     euler_rot = [float(item) for item in euler_rot]
#     location = [float(item) for item in ann['pos']]
#     # pose_dict = {
#     #     "location" : ann['pos'],
#     #     "rotation" : {
#     #         "value" : euler_rot,
#     #     }
#     # }
#     # cam_poses.append(pose_dict)

#     line = f'{location[0]} {location[1]} {location[2]} {euler_rot[0]} {euler_rot[1]} {euler_rot[2]}\n'
#     camear_pose_output_fid.write(line)
# camear_pose_output_fid.close()

# camera_loader_module = [
#     {
#       "module": "camera.CameraLoader",
#       "config": {
#         "path": camera_pose_fname,
#         "file_format": "location rotation/value",
#         "intrinsics": {
#           "cam_K": [376.72453850819767, 0.0, 320.0, 0.0, 376.72453850819767, 240.0, 0.0, 0.0, 1.0],
#           "resolution_x": 640,
#           "resolution_y": 480,
#         }
#       }
#     },
# ]

# object_loader_module = [{
#     'module': 'loader.ObjectLoader',
#     'config': {
#         'paths' : obj_paths,
#     }
# }]
# world_manipulator_module = [
#     {
#     "module": "manipulators.WorldManipulator",
#     "config": {
#         "cf_set_world_category_id": 0  # this sets the worlds background category id to 0
#     }
#     }
# ]
# light_module = [
#     {
#       "module": "lighting.LightLoader",
#       "config": {
#         "lights": [
#           {
#             "type": "POINT",
#             "location": [1, 1, 1],
#             "energy": 1000
#           }
#         ]
#       }
#     }
# ]
# rgb_renderer_module = [
#     {
#       "module": "renderer.RgbRenderer",
#       "config": {
#         "output_key": "colors",
#         "samples": 350,
#         "render_normals": True,
#         "render_depth": True
#       }
#     },
# ]
# write_module = [
#     {
#       "module": "writer.Hdf5Writer",
#       "config": {
#       }
#     }
# ]

# initialize_module = [
#     {
#       "module": "main.Initializer",
#       "config": {
#         "global": {
#           "output_dir": blender_proc_output_dir,
#         }
#       }
#     }
# ]
# modules = initialize_module + object_loader_module 
# modules = modules + light_module 
# modules = modules + camera_loader_module 
# modules = modules + entity_manipulators
# modules = modules + rgb_renderer_module
# modules = modules + write_module

# final_yaml_dict = {
#     "version": 3,
#     "setup": {
#         "blender_install_path": "/home/<env:USER>/blender/",
#         "pip": [
#             "h5py",
#             "imageio"
#         ]
#     },
#     "modules" : modules,
# }


# def create_object_model(ann):
#     '''
#     Args:
#         ann:
#             name, synset_id, model_id, actual_size, position, euler
    
#     Return:
#         output_obj_file: str
#         entity_manipulator: dict
#     '''
#     model_name = ann['name']
#     shapenet_dir = os.path.join(
#         options.shapenet_filepath,
#         '{}/{}/models'.format(ann['synset_id'], ann['model_id']),
#     )
#     input_obj_file = os.path.join(shapenet_dir,'model_normalized.obj')
#     mtl_path = os.path.join(shapenet_dir, 'model_normalized.mtl')

#     model_dir = os.path.join(options.blender_model_root_dir, model_name)
#     if not os.path.exists(model_dir):
#         os.mkdir(model_dir)
#     output_obj_file = os.path.join(model_dir, 'textured.obj')
#     new_mtl_path = os.path.join(model_dir, 'model_normalized.mtl')
#     shutil.copyfile(mtl_path, new_mtl_path)

#     size_x, size_y, size_z = ann['actual_size']
#     bp_utils.scale_obj_file(input_obj_file, output_obj_file, [size_x, size_z, size_y], add_name=model_name)

#     position = [float(item) for item in ann['position']]
#     rot_euler = [float(item) for item in ann['euler']] #[(1/2)*np.pi, 0, 0]
#     config_dict = {
#         "selector" : {
#             "provider" : "getter.Entity",
#             "check_empty": True,
#             "conditions": {
#                 "name": model_name,
#                 "type": "MESH"  # this guarantees that the object is a mesh, and not for example a camera
#             }
#         },
#         "location" : position,
#         "rotation_euler" : rot_euler,
#     }
#     entity_manipulator = {
#         "module": "manipulators.EntityManipulator",
#         "config": config_dict,
#     }

#     return output_obj_file, entity_manipulator
