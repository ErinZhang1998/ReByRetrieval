import os
import json 
import yaml
import shutil
import argparse
import pickle
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

        yaml_dir = os.path.join(output_save_dir, 'yaml_files')
        if not os.path.exists(yaml_dir):
            os.mkdir(yaml_dir)
        self.yaml_dir = yaml_dir

        convex_decomposition_cache_path = os.path.join(self.args.scene_save_dir, "convex_decomposition_cache")
        if not os.path.exists(convex_decomposition_cache_path):
            os.mkdir(convex_decomposition_cache_path)
        self.convex_decomposition_cache_path = convex_decomposition_cache_path
        
        self.config = yaml.load(open(args.blender_proc_config), Loader=yaml.SafeLoader)

        self.camera_intrinsics = {
            "cam_K": [376.72453850819767, 0.0, 320.0, 0.0, 376.72453850819767, 240.0, 0.0, 0.0, 1.0],
            "resolution_x": self.width,
            "resolution_y": self.height,
        }

        self.z_length_dict = {}

    def add_camera_to_scene(self, radius_min=None):
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
        if self.args.single_object:
            middle_z = float(self.config['table']['size'][-1] + self.z_length_dict[0]*0.5)
            high_z = float(self.config['table']['size'][-1] + self.z_length_dict[0])
            x = 0.5 if self.z_length_dict[0] < 0.5 else 0.7
            
            camera_sampler_dict = {
                "module": "camera.CameraLoader",
                "config": {
                    "cam_poses": [
                        {
                            "location": [x, 0, middle_z],
                            "rotation": {
                                "format": "look_at",
                                "value": mean_position_center,
                            }
                        },
                        {
                            "location": [x, 0, float(high_z+0.2)],
                            "rotation": {
                                "format": "look_at",
                                "value": mean_position_center,
                            }
                        },
                    ],
                    "intrinsics": self.camera_intrinsics,
                }
            }
            return [camera_sampler_dict]
        
        camera_config = self.config['camera']
        if radius_min is not None:
            radius_min = radius_min #+ camera_config['radius_gap']
            radius_max = radius_min + camera_config['radius_gap'] #* 2
        else:
            radius_min = camera_config['radius_min']
            radius_max = camera_config['radius_max']
        
        mean_position_shell_location = {
            "provider":"sampler.Shell",
            "center": mean_position_center,
            "radius_min": float(radius_min),
            "radius_max": float(radius_max),
            "elevation_min": float(camera_config['elevation_min']),
            "elevation_max": float(camera_config['elevation_max']),
            "uniform_elevation": True,
        }
        
        camera_sampler_dict = camera_sampler_dict = {
            "module": "camera.CameraSampler",
            "config": {
                "cam_poses": [
                    {
                        "max_tries" : camera_config["max_tries"],
                        "number_of_samples": int(camera_config['number_of_samples']),
                        "location": mean_position_shell_location,
                        "rotation": {
                            "format": "look_at",
                            "value": mean_position_center,
                        },
                        # "check_pose_novelty_rot" : camera_config['check_pose_novelty_rot'],
                        # "min_var_diff_rot" : camera_config['min_var_diff_rot'],
                        # "check_pose_novelty_translation" : camera_config['check_pose_novelty_translation'], 
                        # "min_var_diff_translation" : camera_config['min_var_diff_translation'],
                        # "proximity_checks": {
                        #     "min": camera_config['proximity_checks_min'],
                        #     "max": camera_config['proximity_checks_max'],
                        # },
                        # "excluded_objs_in_proximity_check":  [
                        #     {
                        #         "provider": "getter.Entity",
                        #         "conditions": {
                        #             "cp_physics": False,
                        #         }
                        #     },
                        # ],
                        # "check_if_objects_visible": {
                        #     "provider": "getter.Entity",
                        #     "conditions": {
                        #         "cp_is_object": True,
                        #         "type": "MESH"
                        #     }
                        # },
                    }
                ],
                "intrinsics": self.camera_intrinsics,
            }
        }

        return [camera_sampler_dict]

    def add_lights_to_scene(self):

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
    
    def get_walls(self):
        area_size = self.config['wall_width']
        walls = {
            "module": "constructor.BasicMeshInitializer",
            "config": {
                "meshes_to_add": [
                {
                    "type": "plane",
                    "name": "ground_plane0",
                    "scale": [area_size, area_size, 1],
                    "add_properties": {
                        "cp_model_name": "ground_plane0",
                        "cp_category_id" : 0,
                    },
                    },
                {
                    "type": "plane",
                    "name": "ground_plane1",
                    "scale": [area_size, area_size, 1],
                    "location": [0, -area_size, area_size],
                    "rotation": [-1.570796, 0, 0], # switch the sign to turn the normals to the outside
                    "add_properties": {
                        "cp_model_name": "ground_plane1",
                        "cp_category_id" : 0,
                    },
                    },
                {
                    "type": "plane",
                    "name": "ground_plane2",
                    "scale": [area_size, area_size, 1],
                    "location": [0, area_size, area_size],
                    "rotation": [1.570796, 0, 0],
                    "add_properties": {
                        "cp_model_name": "ground_plane2",
                        "cp_category_id" : 0,
                    },
                    },
                {
                    "type": "plane",
                    "name": "ground_plane3",
                    "scale": [area_size, area_size, 1],
                    "location": [area_size, 0, area_size],
                    "rotation": [0, -1.570796, 0],
                    "add_properties": {
                        "cp_model_name": "ground_plane3",
                        "cp_category_id" : 0,
                    },
                    },
                {
                    "type": "plane",
                    "name": "ground_plane4",
                    "scale": [area_size, area_size, 1],
                    "location": [-area_size, 0, area_size],
                    "rotation": [0, 1.570796, 0],
                    "add_properties": {
                        "cp_model_name": "ground_plane4",
                        "cp_category_id" : 0,
                    },
                    },
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

    def create_table_old(self):
        table_config = self.config['table']
        synset_id = table_config['synset_id']
        if table_config['random']:
            all_available_tables = json.load(open(table_config['table_json']))
            model_id = np.random.choice(all_available_tables)
            table_ann = {
                'synsetId' : synset_id,
                'ShapeNetModelId' : model_id,
                'actual_size' : table_config['size'],
            }
            model_dir_models = os.path.join(
                self.args.blender_model_root_dir,
                '{}/{}/models'.format(table_ann['synsetId'], table_ann['ShapeNetModelId']),
            )
            if table_config['always_overwrite'] or not os.path.exists(model_dir_models):
                bp_utils.save_normalized_object_to_file(
                    self.shapenet_filepath, 
                    self.args.blender_model_root_dir, 
                    table_ann, 
                    actual_size_used=True
                )
            table_path = os.path.join(model_dir_models, 'model_normalized.obj')
        else:
            model_id = table_config['model_id']
            table_path = os.path.join(
                self.args.blender_model_root_dir,
                '{}/{}/models'.format(synset_id, model_id),
                'model_normalized.obj',
            )
        table_module = {
            "module": "loader.ObjectLoader",
            "config": {
                "paths": [table_path],
                "add_properties": {
                    "cp_physics": False,
                    "cp_category_id" : self.num_objects+1,
                    "cp_shape_net_table" : True,
                    "cp_model_name": f'{self.train_or_test}_scene_{self.scene_num}_table',
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
                "location" : [0, 0, float(0.5 * table_config['size'][-1])],
                "rotation_euler" : [float((1/2)*np.pi), 0, 0],
            },
        }
        return table_module, entity_manipulator
    
    def create_object(self, object_idx):
        object_config = self.config['object']
        synset_id = '0{}'.format(self.selected_objects[object_idx]['synsetId'])
        model_id = self.selected_objects[object_idx]['ShapeNetModelId']
        model_name = f'{self.train_or_test}_scene_{self.scene_num}_object_{object_idx}'
        model_dir = os.path.join(self.args.blender_model_root_dir, '{}/{}'.format(synset_id, model_id))
        output_obj_file = os.path.join(model_dir, 'models', 'model_normalized.obj')
        bb_max, bb_min = bp_utils.load_max_min_info(model_dir) 
        if self.args.single_object:
            sampled_xy_size = float(np.random.uniform(self.config['single_object'][synset_id][0], self.config['single_object'][synset_id][1]))
        else:
            sampled_xy_size = float(np.random.uniform(object_config[synset_id][0], object_config[synset_id][1]))
        x_length,z_length,y_length = bb_max - bb_min
        # if self.args.single_object:
        sampled_scale = sampled_xy_size / max(x_length, y_length)

        object_module = {
            "module": "loader.ObjectLoader",
            "config": {
                "path": output_obj_file,
                "add_properties": {
                    "cp_physics": True,
                    "cp_category_id" : object_idx+1,
                    "cp_is_object" : True,
                    "cp_model_name" : model_name,
                }
            }
        }

        x_range, new_z_length, y_range = (bb_max - bb_min) * sampled_scale
        self.z_length_dict[object_idx] = new_z_length
        
        if self.args.single_object:
            entity_manipulator = {
                "module": "manipulators.EntityManipulator",
                "config": {
                    "selector" : {
                        "provider" : "getter.Entity",
                        "check_empty": True,
                        "conditions": {
                            "cp_model_name": model_name,
                            "type": "MESH"  
                        }
                    },
                    "location" : [0,0,float(self.config['table']['size'][-1] + 0.5*new_z_length)],
                    "rotation_euler" : [float((1/2)*np.pi), 0, 0],
                    "scale" : [float(sampled_scale)] * 3,
                },
            }
        else:
            entity_manipulator = {
                "module": "manipulators.EntityManipulator",
                "config": {
                    "selector" : {
                        "provider" : "getter.Entity",
                        "check_empty": True,
                        "conditions": {
                            "cp_model_name": model_name,
                            "type": "MESH"  
                        }
                    },
                    "scale" : [float(sampled_scale)] * 3,
                },
            }

        return object_module, entity_manipulator, x_range, y_range
        
    def get_cc_preload(self):
        texture_to_use = []
        #with open(self.config['texture_to_use_file'], 'rb') as fh:
        with open(self.args.texture_to_use_file, 'rb') as fh:
            texture_to_use = pickle.load(fh)
        cc_preload = [{
            "module": "loader.CCMaterialLoader",
            "config": {
                "used_assets": texture_to_use,
                "folder_path": self.args.cctextures_path,
                "preload": True,
            }
        }]
        return cc_preload
    
    def get_object_material_manipulator(self):
        material_randomization_level = self.config['material_randomization_level'] if not self.args.single_object else 0
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
                    "randomization_level": float(material_randomization_level),
                    "materials_to_replace_with": {
                        "provider": "getter.Material",
                        "random_samples": 1,
                        "conditions": {
                            "cp_is_cc_texture": True 
                        }
                    }
                }
            }
        },
        ]
        return material_manipulator
        
    def object_physics_positioning(self):
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
        return physics_positioning_module
    
    def get_writers(self):
        write_module = [
            {
                "module": "writer.ObjectStateWriter",
                "config" : {
                    "attributes_to_write" : ["customprop_model_name", "customprop_category_id", "name", "location", "rotation_euler", "matrix_world"]
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
        return write_module
    
    def get_object_surface_sampler(self, object_x_ranges=None, object_y_ranges=None):
        on_surface_config = self.config['object_on_surface_sampler']
        if on_surface_config['use_range'] and object_x_ranges is not None and object_y_ranges is not None:
            max_x = np.max(object_x_ranges)
            sum_y = np.sum(object_y_ranges)
            total_area = max_x * sum_y
            range_side_length = float(np.sqrt(total_area))
            all_ranges = np.sort(list(object_x_ranges) + list(object_y_ranges))[::-1]
            largest_range = np.max(all_ranges)
            range_side_length = largest_range * 3 + 0.3
            face_sample_range_min = 0.5 * (1 - (range_side_length / self.config['table']['size'][0]))
            face_sample_range_max = 1 - face_sample_range_min
            max_distance = 0.5 * (all_ranges[0] + all_ranges[1]) * np.sqrt(2) 
        else:
            face_sample_range_min = on_surface_config['face_sample_range_min']
            face_sample_range_max = on_surface_config['face_sample_range_max']
            range_side_length = self.config['table']['size'][0] * (face_sample_range_max - face_sample_range_min)
            max_distance = on_surface_config['max_distance']
        # About rotation:
        if np.random.uniform(0,1) > self.args.upright_ratio:
            max_rotation = [0,0,0]
            min_rotation = [6.28,6.28,6.28]
        else:
            max_rotation = [(1/2)*np.pi,0,0]
            min_rotation = [(1/2)*np.pi,0,6.28]

        object_surface_sampler_module = [{
            "module": "object.OnSurfaceSampler",
            "config": {
                "max_iterations" : on_surface_config['max_iterations'],
                "objects_to_sample": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_physics": True,
                    }
                },
                "surface": {
                    "provider": "getter.Entity",
                    "index": 0,
                    "conditions": {
                       "cp_shape_net_table" : True,
                    }
                },
                "pos_sampler": {
                    "provider": "sampler.UpperRegionSampler",
                    "to_sample_on": {
                        "provider": "getter.Entity",
                        "index": 0,
                        "conditions": {
                            "cp_shape_net_table" : True,
                        }
                    },
                    "min_height": on_surface_config['min_height'],
                    "max_height": on_surface_config['max_height'],
                    "face_sample_range": [float(face_sample_range_min), float(face_sample_range_max)],
                },
                "min_distance": on_surface_config['min_distance'],
                "max_distance": float(max_distance),
                "rot_sampler": {
                    "provider": "sampler.Uniform3d",
                    "max": max_rotation,
                    "min": min_rotation,
                }
            }
        }]

        return object_surface_sampler_module, range_side_length

    def get_world_module(self):
        world_module = [
            {
                "module": "manipulators.WorldManipulator",
                "config": {
                    "cf_set_world_category_id": 0,
                }
            }
        ]
        return world_module
    
    def output_yaml_single_object(self):
        yaml_obj = yaml.load(open(os.path.join(self.output_save_dir, 'standard.yaml')))
        
        modules = yaml_obj['modules']
        pop_idx = None
        intializer_idx = None
        manipulator_idx = None
        camera_sampler_idx = None
        
        for module_idx, module in enumerate(modules):
            if module['module'] == 'camera.CameraSampler':
                camera_sampler_idx = module_idx
            if module['module'] == 'manipulators.EntityManipulator':
                if 'selector' in module['config']:
                    if 'cp_is_object' in module['config']['selector']['conditions']:
                        if module['config']['selector']['conditions']['cp_is_object']:
                            manipulator_idx = module_idx
            if module['module'] == 'main.Initializer':
                intializer_idx = module_idx
            if module['module'] == 'loader.ObjectLoader':

                if 'cp_is_object' in module['config']['add_properties']:
                    if module['config']['add_properties']['cp_is_object']:
                        pop_idx = module_idx
        # _ = modules.pop(pop_idx)

        object_module, object_manipulator, x_range, y_range = self.create_object(0)
        if pop_idx is not None:
            modules[pop_idx] = object_module
        if intializer_idx is not None:
            modules[intializer_idx] = {
                "module": "main.Initializer",
                "config": {
                    "global": {
                    "output_dir": self.scene_folder_path,
                    }
                }
            }
        if manipulator_idx is not None:
            modules[manipulator_idx] = object_manipulator
        
        if camera_sampler_idx is not None:
            modules[camera_sampler_idx] = self.add_camera_to_scene()[0]
        
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
        # print("Output to: ", yaml_fname)
        with open(yaml_fname, 'w+') as outfile:
            yaml.dump(final_yaml_dict, outfile, default_flow_style=False)
        self.yaml_fname = yaml_fname

    
    def output_yaml(self):
        if self.args.single_object:
            self.output_yaml_single_object()
            return
        table_module, table_manipulator = self.create_table_old()
        object_loader_module = [table_module]
        object_manipulator_module = [table_manipulator]
        object_surface_sampler_module = []
        object_x_ranges = []
        object_y_ranges = []
        for object_idx in range(self.num_objects):
            object_module, object_manipulator, x_range, y_range = self.create_object(object_idx)
            object_loader_module += [object_module]
            object_manipulator_module += [object_manipulator]
            object_surface_sampler_module += []
            object_x_ranges += [x_range]
            object_y_ranges += [y_range]

        # object_surface_sampler_module, range_side_length = self.get_object_surface_sampler()
        object_surface_sampler_module, range_side_length = self.get_object_surface_sampler(object_x_ranges, object_y_ranges)

        modules = [
            {
            "module": "main.Initializer",
            "config": {
                "global": {
                "output_dir": self.scene_folder_path,
                }
            }
            }
        ]
        modules += object_loader_module
        modules += object_manipulator_module
        modules += self.get_cc_preload()
        # if not self.args.single_object:
        modules += self.get_object_material_manipulator()

        modules += self.get_walls()
        modules += object_surface_sampler_module
        # modules += self.object_physics_positioning()
        modules += self.get_world_module()
        
        cc_fill_in = [
            {
                "module": "loader.CCMaterialLoader",
                "config": {
                    "folder_path": self.args.cctextures_path,
                    "fill_used_empty_materials": True
                }
            },
        ]
        modules += cc_fill_in
        modules += self.add_lights_to_scene() 
        print("range_side_length: ", range_side_length)
        modules += self.add_camera_to_scene(radius_min=None) 

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
        modules += rgb_renderer_module
        modules += seg_renderer_module
        modules += self.get_writers()
        
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