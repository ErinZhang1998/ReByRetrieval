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
        
        self.config = yaml.load(open(args.blender_proc_config), Loader=yaml.SafeLoader)
        self.all_available_tables = json.load(open(self.config['table_json']))

        self.camera_intrinsics = {
            "cam_K": [376.72453850819767, 0.0, 320.0, 0.0, 376.72453850819767, 240.0, 0.0, 0.0, 1.0],
            "resolution_x": self.width,
            "resolution_y": self.height,
            "fov": 45,
        }

    def add_camera_to_scene(self):
        mean_position_center = {
            "provider": "getter.POI",
            "selector": {
                "provider": "getter.Entity",
                "conditions": {
                    "cp_is_object": True,
                    "type": "MESH"
                }
            }
        }
        # mean_position_part_sphere_location = {
        #     "provider":"sampler.PartSphere",
        #     "center": mean_position_center,
        #     "radius": 1,
        #     "distance_above_center": 0.1,
        #     "mode": "SURFACE"
        # }
        mean_position_shell_location = {
            "provider":"sampler.Shell",
            "center": mean_position_center,
            "radius_min": float(self.config['camera']['radius_min']),
            "radius_max": float(self.config['camera']['radius_max']),
            "elevation_min": float(self.config['camera']['elevation_min']),
            "elevation_max": float(self.config['camera']['elevation_max']),
            "uniform_elevation": True,
        }
        
        camera_sampler_dict = camera_sampler_dict = {
            "module": "camera.CameraSampler",
            "config": {
                "cam_poses": [
                    {
                        "proximity_checks": {
                            "min": 0.3,
                        },
                        "excluded_objs_in_proximity_check":  [
                            {
                                "provider": "getter.Entity",
                                "conditions": {
                                    "cp_physics": False,
                                }
                            },
                            # {
                            #     "provider": "getter.Entity",
                            #     "conditions": {
                            #         "name": "ground_plane.*",
                            #         "type": "MESH",
                            #     }
                            # },
                        ],
                        "number_of_samples": int(self.config['camera']['number_of_samples']),
                        "check_if_objects_visible": {
                            "provider": "getter.Entity",
                            "conditions": {
                                "cp_is_object": True,
                                "type": "MESH"
                            }
                        },
                        "location": mean_position_shell_location,
                        "rotation": {
                            "format": "look_at",
                            "value": mean_position_center,
                        }
                    }
                ],
                "intrinsics": self.camera_intrinsics,
            }
        }

        return [camera_sampler_dict]

    def add_lights_to_scene(self):
        # light_module = {
        #     "module": "lighting.LightLoader",
        #     "config": {
        #         "lights": [
        #             {
        #                 "type": "POINT",
        #                 "location": [1, 1, 1],
        #                 "energy": 1000
        #             },
        #         ]
        #     },
        # }
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
            "radius_min": self.config['light']['radius_min'],
            "radius_max": self.config['light']['radius_max'],
            "elevation_min": self.config['light']['elevation_min'],
            "elevation_max": self.config['light']['elevation_max'],
            "uniform_elevation": True
        }
        
        
        
        min_number_of_samples = self.config['light']['min_number_of_samples']
        max_number_of_samples = self.config['light']['max_number_of_samples']+1
        num_lights = np.random.randint(min_number_of_samples, max_number_of_samples)
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
                            "min": self.config['light']['min_intensity'],
                            "max": self.config['light']['max_intensity'],
                        }
                    }
                ] * num_lights
            }
        }]
        return light_module
    
    def add_random_room(self):
        room_constructor = {
            "module": "constructor.RandomRoomConstructor",
            "config": {
                "floor_area": 100,
                "amount_of_extrusions": 5,
                "used_loader_config": [
                    # {
                    #     "module": "loader.ShapeNetLoader",
                    #     "config": {
                    #         "data_path": self.shapenet_filepath,
                    #         "used_synset_id": "04256520",
                    #     },
                    #     "amount_of_repetitions": 1,
                    # },
                ],
                "add_properties": {
                    "cp_category_id" : 0,
                }
            }
        }
        return [room_constructor]
    
    def add_walls(self):
        area_size = 5
        walls = {
            "module": "constructor.BasicMeshInitializer",
            "config": {
                "meshes_to_add": [
                {
                    "type": "plane",
                    "name": "ground_plane0",
                    "scale": [area_size, area_size, 1]
                    },
                {
                    "type": "plane",
                    "name": "ground_plane1",
                    "scale": [area_size, area_size, 1],
                    "location": [0, -area_size, area_size],
                    "rotation": [-1.570796, 0, 0] # switch the sign to turn the normals to the outside
                    },
                {
                    "type": "plane",
                    "name": "ground_plane2",
                    "scale": [area_size, area_size, 1],
                    "location": [0, area_size, area_size],
                    "rotation": [1.570796, 0, 0]
                    },
                {
                    "type": "plane",
                    "name": "ground_plane3",
                    "scale": [area_size, area_size, 1],
                    "location": [area_size, 0, area_size],
                    "rotation": [0, -1.570796, 0]
                    },
                {
                    "type": "plane",
                    "name": "ground_plane4",
                    "scale": [area_size, area_size, 1],
                    "location": [-area_size, 0, area_size],
                    "rotation": [0, 1.570796, 0]
                    },
                # {
                #     "type": "plane",
                #     "name": "light_plane",
                #     "location": [0, 0, 10],
                #     "scale": [area_size, area_size, 1]
                #     }
                ]
            }
        }
        plane_physics = {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {
                "provider": "getter.Entity",
                "conditions": {
                    "name": '.*plane.*'
                }
                },
                "cp_physics": False,
                "cp_category_id": 0,
            }
        }
        # self.num_objects+2

        wall_material_module = []
        for wall_idx in range(5):
            wall_name = f"ground_plane{wall_idx}"
            wall_material_module.append(
                {
                    "module": "manipulators.EntityManipulator",
                    "config": {
                        "selector": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "name": wall_name,
                        }
                        },
                        "mode": "once_for_all",
                        "cf_randomize_materials": {
                            "randomization_level": 1,
                            "materials_to_replace_with": {
                                "provider": "getter.Material",
                                "random_samples": 1,
                                "conditions": {
                                    "cp_is_cc_texture": True
                                }
                            }
                        }
                    }
                }
            )
        
        return [walls, plane_physics] + wall_material_module
    
    def save_object_to_file(self, ann, actual_size_used=True):
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

        if actual_size_used:
            x,y,z = ann['actual_size']
            bp_utils.scale_obj_file(input_obj_file, output_obj_file, np.array([x,z,y]), add_name=ann['model_name'])
        else:
            bp_utils.normalize_obj_file(input_obj_file, output_obj_file, add_name=ann['model_name'])

        return output_obj_file
    
    def get_model_path(self, ann):
        shapenet_dir = os.path.join(
            self.shapenet_filepath,
            '{}/{}'.format(ann['synset_id'], ann['model_id']),
        )
        input_obj_file = os.path.join(shapenet_dir, 'models', 'model_normalized.obj')
        mtl_path = os.path.join(shapenet_dir, 'models', 'model_normalized.mtl')
        image_material_dir = os.path.join(shapenet_dir, 'images')

        synset_model_dir = os.path.join(self.args.blender_model_root_dir, ann['synset_id'])
        if not os.path.exists(synset_model_dir):
            os.mkdir(synset_model_dir)
        model_dir = os.path.join(synset_model_dir, ann['model_id'])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_dir_models = os.path.join(model_dir, 'models')
        if not os.path.exists(model_dir_models):
            os.mkdir(model_dir_models)
        
        output_obj_file = os.path.join(model_dir_models, 'model_normalized.obj')
        new_mtl_path = os.path.join(model_dir_models, 'model_normalized.mtl')
        new_image_dir = os.path.join(model_dir, 'images')
        if not os.path.exists(new_mtl_path):
            shutil.copyfile(mtl_path, new_mtl_path)

        if os.path.exists(image_material_dir):
            if not os.path.exists(new_image_dir):
                #shutil.rmtree(new_image_dir)
                shutil.copytree(image_material_dir, new_image_dir) 
        else:
            print("WARNING: NO IMAGE DIR FOR TEXTURE", ann)
        
        if not os.path.exists(output_obj_file):
            bp_utils.normalize_obj_file(input_obj_file, output_obj_file)
        
        
        # object_mesh = trimesh.load(output_obj_file, force='mesh')
        # import pdb; pdb.set_trace()
        return output_obj_file
    
    # name, synset_id, model_id
    # actual_size, position, euler
    def create_table(self):
        synset_id = '04379243'
        table_id = np.random.choice(self.all_available_tables)
        table_mesh_fname = os.path.join(self.shapenet_filepath, f'{synset_id}/{table_id}/models/model_normalized.obj')
        table_module = {
            "module": "loader.ShapeNetLoader",
            "config": {
                "data_path": self.shapenet_filepath,
                "used_synset_id": "04379243",
                "add_properties": {
                    "cp_physics": False,
                    "cp_category_id" : self.num_objects+1,
                    "cp_shape_net_table" : True,
                }
            }
        }
        table_rotation = [(1/2)*np.pi,0,0]
        entity_manipulator = {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector" : {
                    "provider" : "getter.Entity",
                    "check_empty": True,
                    "conditions": {
                        "cp_shape_net_table": True,
                        "type": "MESH"  # this guarantees that the object is a mesh, and not for example a camera
                    }
                },
                "location" : [0,0,0],
                "scale" : [3] * 3,
                "rotation_euler" : [float(item) for item in table_rotation],
            },
        }
        return table_module, entity_manipulator

    def create_table_old(self):
        synset_id = '04379243'
        # table_id = '97b3dfb3af4487b2b7d2794d2db4b0e7'
        # table_id = np.random.choice(self.all_available_tables)
        # table_mesh_fname = os.path.join(self.shapenet_filepath, f'{synset_id}/{table_id}/models/model_normalized.obj')
        self.table_info = bp_utils.BlenderProcTable(
            model_name=f'{self.train_or_test}_scene_{self.scene_num}_table',
            synset_id=synset_id,
            model_id_available=self.all_available_tables,
            shapenet_filepath=self.shapenet_filepath,
            num_objects_in_scene=self.num_objects,
            table_size=self.args.table_size,
            table_size_xyz = self.args.table_size_xyz,
        )
        ann = self.table_info.get_blender_proc_dict()
        table_path = self.save_object_to_file(ann)
        table_module = {
            "module": "loader.ObjectLoader",
            "config": {
                "paths": [table_path],
                "add_properties": {
                    "cp_physics": False,
                    "cp_category_id" : self.num_objects+1,
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
                        "cp_shape_net_table": True,
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
        model_name = f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}'
        actual_size = self.selected_objects[object_idx]['size']
        scale = self.selected_objects[object_idx]['scale']
        
        # shapnet_file_name = os.path.join(self.shapenet_filepath, f'{synset_id}/{model_id}/models/model_normalized.obj')
        # object_info = bp_utils.BlenderProcNonTable(
        #     model_name=model_name,
        #     synset_id=synset_id,
        #     model_id=model_id,
        #     shapenet_file_name=shapnet_file_name,
        #     num_objects_in_scene=self.num_objects,
        #     table_size=self.args.table_size,
        #     object_idx=object_idx,
        #     selected_object_info=self.selected_objects[object_idx],
        #     table_height=self.table_info.height,
        #     upright_ratio=self.args.upright_ratio,
        # )
        # self.object_info_dict[object_idx] = object_info
        
        ann = {
            'synset_id' : synset_id,
            'model_id' : model_id,
            'model_name' : model_name,
            'actual_size' : actual_size,
        }
        # ann = object_info.get_blender_proc_dict()
        output_obj_file = self.save_object_to_file(ann, actual_size_used=False)
        # output_obj_file = self.get_model_path(ann)
        object_module = {
            "module": "loader.ObjectLoader",
            "config": {
                "path": output_obj_file,
                "add_properties": {
                    "cp_physics": True,
                    "cp_category_id" : object_idx+1,
                    "cp_is_object" : True,
                    "cp_model_name" : ann['model_name'],
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
                        "name" : ann['model_name'],
                        "cp_model_name": ann['model_name'],
                        "type": "MESH"  # this guarantees that the object is a mesh, and not for example a camera
                    }
                },
                "scale" : [float(item) for item in scale]
                # "location" : [float(item) for item in ann['position']],
                # "rotation_euler" : [float(item) for item in ann['euler']],
            },
        }

        # About rotation:
        if np.random.uniform(0,1) > self.args.upright_ratio:
            max_rotation = [0,0,0]
            min_rotation = [6.28,6.28,6.28]
        else:
            max_rotation = [(1/2)*np.pi,0,0]
            min_rotation = [(1/2)*np.pi,0,6.28]
        
        face_sample_range_min = self.config['object_on_surface_sampler']['face_sample_range_min']
        face_sample_range_min = float(face_sample_range_min)
        face_sample_range_max = self.config['object_on_surface_sampler']['face_sample_range_max']
        face_sample_range_max = float(face_sample_range_max)

        # surface_pose_sampler = {
        #     "module": "object.OnSurfaceSampler",
        #     "config": {
        #         "objects_to_sample": {
        #             "provider": "getter.Entity",
        #             "conditions": {
        #                 "cp_model_name": model_name
        #             }
        #         },
        #         "surface": {
        #             "provider": "getter.Entity",
        #             "index": 0,
        #             "conditions": {
        #                "cp_shape_net_table": True,
        #                "type": "MESH",
        #             }
        #         },
        #         "pos_sampler": {
        #             "provider": "sampler.UpperRegionSampler",
        #             "to_sample_on": {
        #                 "provider": "getter.Entity",
        #                 "index": 0,
        #                 "conditions": {
        #                     "cp_shape_net_table": True,
        #                     "type": "MESH",
        #                 }
        #             },
        #             "min_height": 0.1,
        #             "max_height": 0.2,
        #             "face_sample_range": [face_sample_range_min,face_sample_range_max],
        #         },
        #         "min_distance": 0.1,
        #         "max_distance": 0.2,
        #         "rot_sampler": {
        #             "provider": "sampler.Uniform3d",
        #             "max": max_rotation,
        #             "min": min_rotation,
        #         }
        #     }
        # }

        
        # 
        return object_module, None, entity_manipulator
        
    def output_yaml(self):
        # table_module, table_manipulator = self.create_table_old()
        object_loader_module = []
        object_manipulator_module = []
        object_surface_sampler_module = []
        for object_idx in range(len(self.selected_objects)):
            object_module, surface_pose_sampler, object_manipulator = self.create_object(object_idx)
            object_loader_module += [object_module]
            object_manipulator_module += [object_manipulator]
            object_surface_sampler_module += [surface_pose_sampler]
        

        # About rotation:
        if np.random.uniform(0,1) > self.args.upright_ratio:
            max_rotation = [0,0,0]
            min_rotation = [6.28,6.28,6.28]
        else:
            max_rotation = [(1/2)*np.pi,0,0]
            min_rotation = [(1/2)*np.pi,0,6.28]
        
        face_sample_range_min = self.config['object_on_surface_sampler']['face_sample_range_min']
        face_sample_range_min = float(face_sample_range_min)
        face_sample_range_max = self.config['object_on_surface_sampler']['face_sample_range_max']
        face_sample_range_max = float(face_sample_range_max)
        object_surface_sampler_module = [{
            "module": "object.OnSurfaceSampler",
            "config": {
                "objects_to_sample": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_physics": True
                    }
                },
                "surface": {
                    "provider": "getter.Entity",
                    "index": 0,
                    "conditions": {
                       "name": "ground_plane0"
                    }
                },
                "pos_sampler": {
                    "provider": "sampler.UpperRegionSampler",
                    "to_sample_on": {
                        "provider": "getter.Entity",
                        "index": 0,
                        "conditions": {
                            "name": "ground_plane0"
                        }
                    },
                    "min_height": 0.5,
                    "max_height": 1,
                    "face_sample_range": [0.4, 0.6],
                },
                "min_distance": 0.01,
                "max_distance": 0.20,
                "rot_sampler": {
                    "provider": "sampler.Uniform3d",
                    "max": max_rotation,
                    "min": min_rotation,
                }
            }
        }
        ]

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
        seg_renderer_module = [
            {
                "module": "renderer.SegMapRenderer",
                "config": {
                    "map_by": ["instance", "class", "name"],
                    "default_values": {
                        "class": 0,
                        "cp_category_id" : 0,
                    },
                }
            },
        ]
        write_module = [
            {
                "module": "writer.ObjectStateWriter",
                "config" : {
                    "attributes_to_write" : ["name", "location", "rotation_euler", "matrix_world"]
                }
            },
            {
                "module": "writer.LightStateWriter"
            },
            {
                "module": "writer.CameraStateWriter",
                "config": {
                    "attributes_to_write": ["location", "rotation_euler", "fov_x", "fov_y", "shift_x", "shift_y", "cam_K", "cam2world_matrix"]
                }
            },
            {
                "module": "writer.CocoAnnotationsWriter",
                "config": {
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
        additional_light_module = [{
            "module": "lighting.SurfaceLighting",
            "config": {
                "selector": {
                "provider": "getter.Entity",
                "conditions": {
                    "name": "Ceiling"
                },
                "emission_strength": 4.0
                }
            }
        }]
        world_module = [
            {
                "module": "manipulators.WorldManipulator",
                "config": {
                    "cf_set_world_category_id": self.num_objects+1,
                }
            }
        ]
        cc_preload = [{
            "module": "loader.CCMaterialLoader",
            "config": {
                # "used_assets": [
                #     "Bricks", 
                #     "Wood", 
                #     "Carpet", 
                #     "Tiles", 
                #     "Marble",
                #     "OfficeCeiling",
                #     "Concrete",
                # ],
                "folder_path": self.args.cctextures_path,
                "preload": True,
            }
        }]
        material_manipulator = [{
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "type": "MESH"
                    }
                },
                "cf_randomize_materials": {
                    "randomization_level": float(self.config['material_randomization_level']),
                    "materials_to_replace_with": {
                        "provider": "getter.Material",
                        "random_samples": 1,
                        "conditions": {
                            "cp_is_cc_texture": True  # this will return one random loaded cc textures
                        }
                    }
                }
            }
        },
        ]
        cc_fill_in = [
            {
                "module": "loader.CCMaterialLoader",
                "config": {
                    "folder_path": self.args.cctextures_path,
                    "fill_used_empty_materials": True
                }
            },
        ]
        wall_material_module = [{
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {
                "provider": "getter.Entity",
                "conditions": {
                    "name": "ground_plane.*"
                }
                },
                "mode": "once_for_all",
                "cf_randomize_materials": {
                    "randomization_level": 1,
                    "materials_to_replace_with": {
                        "provider": "getter.Material",
                        "random_samples": 1,
                        "conditions": {
                            "cp_is_cc_texture": True
                        }
                    }
                }
            }
        }]
        modules = initialize_module 
        # modules += [table_module]
        modules += object_loader_module
        # modules += [table_manipulator]
        modules += object_manipulator_module

        modules += cc_preload
        modules += material_manipulator
        # modules += self.add_random_room()
        modules += self.add_walls()
        


        modules += object_surface_sampler_module
        modules += physics_positioning_module
        modules += world_module
        # modules += cc_preload
        # modules += material_manipulator
        # # modules += self.add_random_room()
        # modules += self.add_walls()
        # # modules += wall_material_module
        modules += cc_fill_in
        modules += light_module 
        # modules += additional_light_module
        modules += camera_module 
        modules += rgb_renderer_module
        modules += seg_renderer_module
        modules += write_module
        
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
        self.yaml_fname = yaml_fname


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
