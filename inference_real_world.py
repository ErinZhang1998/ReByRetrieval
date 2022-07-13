from all_imports import *
from all_detectron_imports import *

parser = argparse.ArgumentParser()
parser.add_argument("--inference_config_yaml", dest="inference_config_yaml")
parser.add_argument("--retrieval_config_file", dest="retrieval_config_file")
parser.add_argument("--redo", dest="redo", action="store_true")

def save_depth_as_png(depth, depth_path):
    depth_scaled = depth * 1000
    depth_scaled[depth_scaled > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
    depth_img = depth_scaled.astype(np.uint16)
    cv2.imwrite(depth_path, depth_img)

def filter_mask_by_depth_outlier(depth, mask):
    object_mask = copy.deepcopy(mask)
    masked_depth = depth * object_mask 
    # import pdb; pdb.set_trace()
    min_depth = np.percentile(masked_depth[masked_depth > 0], 5)
    max_depth = np.percentile(masked_depth[masked_depth > 0], 92)
    
    indices = np.vstack(np.where(masked_depth < min_depth))
    object_mask[tuple(indices)] = 0
    indices = np.vstack(np.where(masked_depth > max_depth))
    object_mask[tuple(indices)] = 0

    return object_mask

def update_yaml(options, yaml_obj):
    with open(options.inference_config_yaml, 'w') as outfile:
        yaml.dump(yaml_obj, outfile, default_flow_style=False, sort_keys=False)

def intrinsics_scale(scale, width, height, fx, fy, cx, cy):
        if scale == 1:
            return fx, fy, cx, cy
        center_x = float(width - 1) / 2
        center_y = float(height - 1) / 2
        orig_cx_diff = cx - center_x
        orig_cy_diff = cy - center_y
        height = scale * height
        width = scale * width
        scaled_center_x = float(width - 1) / 2
        scaled_center_y = float(height - 1) / 2
        fx = scale * fx
        fy = scale * fy
        # skew = scale * skew
        cx = scaled_center_x + scale * orig_cx_diff
        cy = scaled_center_y + scale * orig_cy_diff

        return fx, fy, cx, cy

def detectron(args, paths):
    images_pil_orig = [
        PIL.Image.fromarray(np.load(os.path.join(path[1], path[2])).astype(np.uint8)) for path in paths
    ]
    images = [
        np.asarray(img.resize((args.detectron.width, args.detectron.height))) for img in images_pil_orig
    ]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = args.detectron.inference.model_path
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cat_utils.shapenet_category_idx_to_name)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = t_det.Predictor(cfg)
    MetadataCatalog.get(args.detectron.test.dataset_name).thing_classes = list(cat_utils.shapenet_category_idx_to_name.values())
    predictions = predictor(images)

    # json_objs = {}
    json_objs = []
    for idx, (image_idx, parent_dir, rgb_path, depth_path) in enumerate(paths):
        # parent_dir = os.path.abspath(os.path.join(rgb_path, os.pardir))
        rgb_path = os.path.join(parent_dir, rgb_path)
        depth_path = os.path.join(parent_dir, depth_path)
        mask_path_template = os.path.join(parent_dir, 'segmentation_{}_orig.png')
        
        masks = predictions[idx]['instances'].to('cpu').pred_masks.float().numpy()
        bbox = predictions[idx]['instances'].to('cpu').pred_boxes.tensor.float().numpy()
        
        if rgb_path.endswith('.npy'):
            png_path = rgb_path.replace('.npy', '_orig.png')
        else:
            png_path = rgb_path.replace('.png', '_orig.png')
        images_pil_orig[idx].save(png_path)
        
        for mask_idx, mask in enumerate(masks):
            seg_save_path_full = mask_path_template.format(mask_idx)
            cv2.imwrite(seg_save_path_full, mask.astype(np.uint8))
            json_obj = {
                'image_idx' : image_idx,
                'image_path' :  png_path,
                'depth_path' : depth_path,
                'mask_idx' : mask_idx,
                'mask_file_path' : seg_save_path_full,
                'bbox' : [float(elem) for elem in bbox[mask_idx]],
            }
            # print(json_obj)
            # json_objs[(image_idx, mask_idx)] = json_obj
            json_objs.append(json_obj)
    
    return json_objs    

def retrieval(args, inf_yaml):
    json_objs_vals = json.load(open(inf_yaml['detectron']['output_file']))
    json_objs = {(val['image_idx'], val['mask_idx']) : val for val in json_objs_vals}

    retrieval_model = PretrainedResNetSpatialSoftmax(args)
    retrieval_model = retrieval_model.cuda(device=torch.cuda.current_device())
    uu.load_model_from(args, retrieval_model, data_parallel=False)

    w,h = args.dataset_config.size_w, args.dataset_config.size_h

    cam_extrinsics = np.asarray(args.real_world.camera_extrinsics)
    if args.model_config.extrinsics_in_dim == 12:
        extrinsics = cam_extrinsics.reshape(1, -1)
    elif args.model_config.extrinsics_in_dim == 6:
        camera_position = cam_extrinsics[:,-1]
        camera_euler = R.from_matrix(cam_extrinsics[:,:-1]).as_euler('xyz')
        extrinsics = np.concatenate([camera_position, camera_euler]).reshape(1, -1)
    elif args.model_config.extrinsics_in_dim == 7:
        camera_position = cam_extrinsics[:,-1]
        camera_quat = R.from_matrix(cam_extrinsics[:,:-1]).as_quat()
        extrinsics = np.concatenate([camera_position, camera_quat]).reshape(-1,)
    
    # prepare input 
    image_inputs = []
    sample_ids = []
    for (image_idx, mask_idx), json_obj in json_objs.items():
        rgb_path = json_obj['image_path']
        img_resized = PIL.Image.open(rgb_path).resize((w,h))
        mask = cv2.imread(json_obj['mask_file_path'])[:,:,0]
        mask_resized = utrans.mask_to_PIL(mask).resize((w,h))

        img = torchvision.transforms.ToTensor()(img_resized)
        mask = torchvision.transforms.ToTensor()(mask_resized)
        if len(mask.shape) > 2:
            mask = mask[:1,:,:]
        depth = np.load(json_obj['depth_path'])
        depth = cv2.resize(depth, (w,h), interpolation = cv2.INTER_AREA)
        if args.use_depth:
            img_depth = torch.unsqueeze(torch.FloatTensor(depth), axis=0)
            img_input = torch.cat([img, img_depth, mask], 0)
        else:
            img_input = torch.cat((img, mask), 0)
            
        image_inputs.append(torch.FloatTensor(img_input))
        sample_ids.append([image_idx, mask_idx])
    
    input_tensors = []
    input_extrinsics = []
    batch_size = args.real_world.inference.batch_size
    for batch_idx in range(np.ceil(len(image_inputs) / batch_size).astype(int)):
        start_idx = batch_idx * batch_size 
        end_idx = (batch_idx+1) * batch_size
        if end_idx > len(image_inputs):
            end_idx = len(image_inputs)
        stacked_tensor = torch.stack(image_inputs[start_idx:end_idx], dim=0)
        if args.model_config.condition_on_extrinsics:
            extrinsics_tensor = torch.FloatTensor(np.repeat(extrinsics, len(stacked_tensor), axis=0))
            input_extrinsics.append(extrinsics_tensor)
        input_tensors.append(stacked_tensor)
    
    acc_dict = dict()
    retrieval_model.eval()
    with torch.no_grad():
        for batch_idx in range(len(input_tensors)):
            image_data = input_tensors[batch_idx]
            image_data = image_data.cuda(non_blocking=args.cuda_non_blocking)

            if args.model_config.condition_on_extrinsics:
                extrinsics = input_extrinsics[batch_idx].cuda(non_blocking=args.cuda_non_blocking)
                return_keys, return_vals = retrieval_model([image_data, extrinsics])
            else:
                return_keys, return_vals = retrieval_model([image_data])
        
            for key_idx, key in enumerate(return_keys):
                val = return_vals[key_idx].detach().cpu()
                l = acc_dict.get(key, [])
                l.append(val)
                acc_dict[key] = l
    
    return acc_dict, sample_ids

def prepare_perch_annotations(args, inf_yaml):
    width = inf_yaml['perch']['width']
    height = inf_yaml['perch']['height']
    kinect = inf_yaml['kinect']
    
    target_sample_ids = inference.get_sample_ids(inf_yaml['target']['save_dir'], inf_yaml['target']['epoch'])
    target_sample_ids = np.asarray(target_sample_ids).reshape(-1,)

    sorted_target_idx = np.load(inf_yaml['sorted_target_idx_path'])
    selected_idx = np.load(inf_yaml['selected_idx_path'])
    sorted_target_idx = selected_idx[sorted_target_idx]
    selected_target_sample_id = np.asarray(target_sample_ids)[sorted_target_idx]

    pred_scale = inference.get_features(inf_yaml['query']['save_dir'], inf_yaml['query']['epoch'], fname_template = '{}_scale_pred.npy').reshape(-1,)
    query_sample_id_dir = os.path.join(inf_yaml['query']['save_dir'], 'predictions', '{}_sample_id.npy'.format(inf_yaml['query']['epoch']))
    query_sample_id = np.load(query_sample_id_dir).astype(int)

    fx, fy, cx, cy = intrinsics_scale( width/ kinect['width'], kinect['width'], kinect['height'], kinect['fx'], kinect['fy'], kinect['cx'], kinect['cy'])
    intrinsics_matrix = [
        [float(fx), 0, cx],
        [0, float(fy), cy],
        [0, 0, 1],
    ]
    
    json_objs_vals = json.load(open(inf_yaml['detectron']['output_file']))
    json_objs = {(val['image_idx'], val['mask_idx']) : val for val in json_objs_vals}

    image_idx_to_image_ann = {}
    
    for (image_idx, mask_idx), json_obj in json_objs.items():
        rgb_path = json_obj['image_path']
        rgb_path_new = rgb_path.replace('_orig.png', '.png')
        json_obj['image_path'] = rgb_path_new
        
        if image_idx not in image_idx_to_image_ann:
            img_resized = PIL.Image.open(rgb_path).resize((width,height))
            img_resized.save(rgb_path_new)
            depth = np.load(json_obj['depth_path'])
            depth = cv2.resize(depth, (width, height), interpolation = cv2.INTER_AREA)
            save_depth_as_png(depth, rgb_path_new.replace('rgb', 'depth'))
        
        json_obj['depth_path'] = rgb_path_new.replace('rgb', 'depth')
        if image_idx in image_idx_to_image_ann:
            continue

        perch_root_idx = rgb_path_new.index(inf_yaml['perch_root_dir'])
        rgb_suffix = rgb_path_new[perch_root_idx+1+len(inf_yaml['perch_root_dir']):]
        assert rgb_suffix.endswith('.png')
        image_ann = {
            'id': 0,
            'file_name': rgb_suffix,
            'width': int(width),
            'height': int(height),
            'date_captured': '2021-08-27 23:33:44.336324',
            'license': 1,
            'coco_url': '',
            'flickr_url': '',
            'intrinsics_matrix': intrinsics_matrix,
        }
        image_idx_to_image_ann[image_idx] = image_ann
    
    json_paths = []
    for query_idx, (image_idx, mask_idx) in enumerate(query_sample_id):
        json_obj = json_objs[image_idx, mask_idx]
        x1,y1,x2,y2 = json_obj['bbox']
        x1 *= (width / args.detectron.width)
        x2 *= (width / args.detectron.width)
        y1 *= (height / args.detectron.height)
        y2 *= (height / args.detectron.height)
        bbox = BoxMode.convert(
            np.asarray([x1,y1,x2,y2]).reshape(-1,4),
            BoxMode.XYXY_ABS,
            BoxMode.XYWH_ABS
        )

        scene_dir = os.path.abspath(os.path.join(json_obj['image_path'], os.pardir))
        annotation_dir = os.path.join(scene_dir, inf_yaml['experiment_name'])
        uu.create_dir(annotation_dir)

        for prediction_idx in range(10):
            target_sample_id = selected_target_sample_id[query_idx][prediction_idx] 
            new_model_name = '{}-{}__{}__{}'.format(image_idx, mask_idx, target_sample_id, prediction_idx)
            target_scene_num, target_image, target_category_id = inference.sample_id_to_parts(target_sample_id)
            
            target_annotation_path = os.path.join(
                inf_yaml['target']['data_dir'], 
                f'scene_{target_scene_num:06}', 
                'annotations.json',
            )
            target_json_obj = p_utils.COCOSelf(target_annotation_path)
            target_ann = target_json_obj.category_id_to_ann[target_category_id]
            
            mesh_file_name = os.path.join(
                inf_yaml['blender_proc_model_dir'], 
                target_ann['synset_id'],
                target_ann['model_id'],
                'models',
                'model_normalized.obj',
            )

            if 'use_gt_size' in inf_yaml['perch'] and inf_yaml['perch']['use_gt_size']:
                if 'gt_size' not in json_obj:
                    print("Please annotate!")
                    raise
                scale = None
                size = json_obj['gt_size']
            elif 'use_target_scale' in inf_yaml and inf_yaml['perch']['use_target_scale']:
                scale = target_ann['size']
                size = [-1] * 3
            else:
                if pred_scale[query_idx] < 0:
                    import pdb; pdb.set_trace()
                scale = [pred_scale[query_idx]] * 3
                size = [-1] * 3
                
            new_mesh, scale_xyz = datagen_utils.save_correct_size_model(
                inf_yaml['perch_model_dir'], 
                new_model_name, 
                size, 
                mesh_file_name, 
                scale = scale,
                turn_upright_before_scale = False,
                turn_upright_after_scale = True,
            )
            actual_size = new_mesh.bounds[1] - new_mesh.bounds[0]
            category_ann = {
                'id': 0,
                'supercategory': 'coco_annotations',
                'name': new_model_name,
                'synset_id': target_ann['synset_id'],
                'model_id': target_ann['model_id'],
                'size': [float(item) for item in scale_xyz],
                'actual_size': [float(item) for item in actual_size],
                'half_or_whole': 0,
                'perch_rot_angle': 0,
            }
            image_ann = image_idx_to_image_ann[image_idx]

            mask = cv2.imread(json_obj['mask_file_path'])
            mask_resized = cv2.resize(mask, (width, height), interpolation = cv2.INTER_AREA)[:,:,0]
            mask_resized[mask_resized > 0] = 1
            depth_scaled = cv2.imread(json_obj['depth_path'], cv2.IMREAD_ANYDEPTH)
            # mask_resized = filter_mask_by_depth_outlier(depth_scaled / 1000, mask_resized)
            json_obj['mask_file_path'] = json_obj['mask_file_path'].replace('_orig.png', '.png')
            cv2.imwrite(json_obj['mask_file_path'], mask_resized.astype(np.uint8))
            mask_path = json_obj['mask_file_path']

            perch_root_idx = mask_path.index(inf_yaml['perch_root_dir'])
            mask_suffix = mask_path[perch_root_idx+1+len(inf_yaml['perch_root_dir']):]            
            ann = {
                'id': 0,
                'image_id': 0,
                'category_id': 0,
                'iscrowd': 0,
                'bbox': [int(elem) for elem in bbox.reshape(-1,)],
                'segmentation': None,
                'width': image_ann['width'],
                'height': image_ann['height'],
                'center': None,
                'model_name': new_model_name,
                'mask_file_path': mask_suffix,
            }
            # print(ann['bbox'])
            json_dict = {
                'info' : None,
                'licenses' : None,
                'images' : [image_ann],
                'categories' : [category_ann],
                'annotations' : [ann],
            }
            json_string = json.dumps(json_dict)
            json_path = os.path.join(annotation_dir, f'{new_model_name}.json')
            # print("Json path of {}".format(json_path))
            json_file = open(json_path, 'w+')
            json_file.write(json_string)
            json_file.close()

            json_paths.append(json_path)
    return json_paths

def prepare_perch(args, inf_yaml):
    from datetime import date

    json_paths = None
    with open(inf_yaml['all_json_paths_save_file'], 'rb') as fh:
        json_paths = pickle.load(fh)

    L_perch = {}

    for run_idx, json_path in enumerate(json_paths):
        perch_run_idx = run_idx % inf_yaml['number_of_perch_runs']
        json_path_docker = json_path.replace(inf_yaml['perch_root_dir'], '/data/custom_dataset')

        json_path_docker_parts = json_path_docker.split('/')[-1].split('.')
        if len(json_path_docker_parts) < 2 or json_path_docker_parts[1] != 'json':
            continue
        output_subdir = json_path_docker_parts[0]

        L = L_perch.get(perch_run_idx, [])
        L += [(output_subdir, json_path_docker)]
        L_perch[perch_run_idx] = L
    
    from datetime import date

    perch_run_paths = []
    today = date.today()

    for perch_run_idx, L in L_perch.items():
        perch_annotation_list_path = os.path.join(
            inf_yaml['perch_root_dir'], 
            'perch_pickles',
            '{}_{}_0{}.pickle'.format(today, inf_yaml['experiment_name'], perch_run_idx+1),
        )
        
        fh = open(perch_annotation_list_path, 'wb+')
        pickle.dump(L, fh)
        perch_run_paths += [perch_annotation_list_path]
        fh.close()
    
    perch_model_dir_docker = inf_yaml['perch_model_dir'].replace(inf_yaml['perch_root_dir'], '/data/custom_dataset')
    
    available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    available_devices = [int(elem) for elem in available_devices]

    perch_run_commands_file = inf_yaml['all_json_paths_save_file'].split('.')[0] + '.sh'
    output_fh = open(perch_run_commands_file, 'w+', encoding='utf-8')
    for path_idx, path in enumerate(perch_run_paths):
        device_number = available_devices[path_idx % (len(available_devices))]
        path_in_container = path.replace(inf_yaml['perch_root_dir'], '/data/custom_dataset')
        
        str1 = f"CUDA_VISIBLE_DEVICES={device_number}" + \
        " python fat_pose_image_custom.py" + \
        f" --ros_node_name_suffix 0{path_idx}" + \
        " --config config_custom_docker_real_world.yaml" + \
        f" --model_dir {perch_model_dir_docker}" + \
        " --perch_debug_dir /data/custom_dataset/perch_output/{}".format(inf_yaml['experiment_name']) + \
        " --python_debug_dir /data/custom_dataset/model_output/{}".format(inf_yaml['experiment_name']) + \
        f" --annotation_list_path {path_in_container}" 
        print(str1, "\n")
        line = '{}\n'.format(str1)
        output_fh.write(line)
    output_fh.close()
    return perch_run_commands_file

def main(options):
    inf_yaml = yaml.safe_load(open(options.inference_config_yaml))
    if options.retrieval_config_file is not None:
        inf_yaml['retrieval_config_file'] = options.retrieval_config_file
    
    args = uu.Struct(yaml.safe_load(open(inf_yaml['retrieval_config_file']))) 

    experiment_save_dir = os.path.abspath(os.path.join(args.model_config.model_path, os.pardir, os.pardir))
    experiment_save_dir = os.path.join(experiment_save_dir, inf_yaml['target']['data_name'])
    result_name = args.model_config.model_path.split('/')[-3]
    inf_yaml['result_name'] = result_name
    epoch = int(args.model_config.model_path.split('/')[-1].split('.')[0])

    paths = None
    with open(inf_yaml['query']['image_paths_file'], 'r') as fh:
        lines = fh.readlines()
        paths = [line.strip('\n').split(' ') for line in lines]
        paths = [(int(image_idx), parent_dir, rgb_path, depth_path) for image_idx,parent_dir,rgb_path,depth_path in paths]
    
    if inf_yaml['detectron']['output_file'] is None:
        json_objs = detectron(args, paths)
        json_string = json.dumps(json_objs)
        json_file = os.path.join(os.path.abspath(os.path.join(inf_yaml['query']['image_paths_file'], os.pardir)), 'detectron_output.json')
        json_file_fh = open(json_file, 'w+')
        json_file_fh.write(json_string)
        json_file_fh.close()
        inf_yaml['detectron']['output_file'] = json_file
        update_yaml(options, inf_yaml)

    if (inf_yaml['query']['epoch'] is None or inf_yaml['query']['save_dir'] is None) and not options.redo:
        acc_dict, sample_ids = retrieval(args, inf_yaml)
        query_save_dir = os.path.join(inf_yaml['retrieval']['output_folder'], result_name)
        uu.create_dir(query_save_dir)
        prediction_dir = os.path.join(query_save_dir, 'predictions')
        uu.create_dir(prediction_dir)

        fname = os.path.join(prediction_dir, f'{epoch}_sample_id.npy')
        np.save(fname, np.asarray(sample_ids))
        
        for key in acc_dict.keys():
            value = torch.cat(acc_dict[key], dim=0) 
            value = value.numpy()
            fname = os.path.join(prediction_dir, f'{epoch}_{key}.npy')
            np.save(fname, value)
        
        inf_yaml['query']['epoch'] = epoch
        inf_yaml['query']['save_dir'] = query_save_dir
        update_yaml(options, inf_yaml)

    # Extract target features
    if not os.path.exists(experiment_save_dir) or not os.path.exists(os.path.join(experiment_save_dir, 'predictions')):
        print("PLEASE PROCESS TARGET FEATURES FIRST!!")
        print(' '.join([
            'CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py',
            '--config_file {}'.format(inf_yaml['retrieval_config_file']),
            '--init_method tcp://localhost:2010',
            '--only_test',
            '--feature_extract',
            '--calculate_triplet_loss False',
            '--only_test_epoch {}'.format(epoch),
            '--experiment_save_dir {}'.format(experiment_save_dir),
            '--testing_scene_dir {}'.format(inf_yaml['target']['data_dir']),
            '--testing_yaml_file_dir {}'.format(inf_yaml['target']['yaml_file_root_dir']),
        ]))
        return
    
    inf_yaml['target']['save_dir'] = experiment_save_dir
    inf_yaml['target']['epoch'] = epoch
    update_yaml(options, inf_yaml)

    # Rank target samples for retrieval
    experiment_name_suffix = input("Enter experiment suffix :")
    experiment_name = 'real_world-{}-{}-{}-{}'.format(result_name, inf_yaml['target']['data_name'], epoch, experiment_name_suffix)
    inf_yaml['experiment_name'] = experiment_name
    sorted_target_idx_path, selected_idx_path = inference.output_retrieval_results(uu.Struct(inf_yaml), uniform=True, experiment_name=experiment_name)
    
    inf_yaml['sorted_target_idx_path'] = sorted_target_idx_path
    inf_yaml['selected_idx_path'] = selected_idx_path
    update_yaml(options, inf_yaml)

    # Prepare for perch runs
    json_paths = prepare_perch_annotations(args, inf_yaml)
    all_json_paths_save_file = os.path.join(
        inf_yaml['perch_root_dir'], 
        'perch_annotations_{}.pkl'.format(experiment_name),
    )
    with open(all_json_paths_save_file, 'wb+') as fh:
        pickle.dump(json_paths, fh)
    inf_yaml['all_json_paths_save_file'] = all_json_paths_save_file
    update_yaml(options, inf_yaml)

    perch_run_commands_file = prepare_perch(args, inf_yaml)
    inf_yaml['perch_run_commands_file'] = perch_run_commands_file
    update_yaml(options, inf_yaml)

if __name__ == '__main__':
    options = parser.parse_args()
    main(options)