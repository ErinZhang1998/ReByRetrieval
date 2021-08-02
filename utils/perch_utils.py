import numpy as np 

def process_one_scene_perch_result( 
        df, 
        threshold = 0.02,
        custom = True,
        model_name_to_category_list_name = None,
    ):

    def isnumber(x):
        try:
            float(x)
            return float(x)
        except:
            return -1

    def isNan(x):
            if np.isnan(x):
                return -1
            else:
                return x

    cleaned_df_col = {}
    for col in df:
        df_col = df[col] 
        df_col = df_col.map(isnumber).map(isNan)
        cleaned_df_col[col] = df_col
    
    # Calculate accuracy using threshold
    acc_dict = {}
    
    above_threshold_image_dict = {}
    missing_object_image_dict = {} # model_full_name --> [image_name]
    
    for col_name, df_col in cleaned_df_col.items():
        if col_name.split('-')[-1] != 'adds':
            continue
        model_name = col_name.split('-')[0]
        # print(model_name)
        if model_name_to_category_list_name is not None:
            model_full_name = model_name_to_category_list_name[model_name]
        else:
            model_full_name = model_name
        if custom:
                object_idx = model_name.split('_')[-1]
        col_arr = df_col.to_numpy().astype(float)
        
        for index, img_name in df['name'][df_col > threshold].iteritems():
            image_name_dir = above_threshold_image_dict.get(model_full_name, [])
            if custom:
                image_name_dir.append((img_name, df_col[index], object_idx))   
            else:
                image_name_dir.append((img_name, df_col[index]))
            above_threshold_image_dict[model_full_name] = image_name_dir
        
        for index, img_name in df['name'][df_col < 0].iteritems():
            image_name_dir = missing_object_image_dict.get(model_full_name, [])
            if custom:
                image_name_dir.append((img_name, object_idx))
            else:
                image_name_dir.append(img_name)
            missing_object_image_dict[model_full_name] = image_name_dir

        leng = np.sum((col_arr >= 0))
        if leng == 0:
            continue
        L, total_num = acc_dict.get(model_full_name, ([], 0))
        L += list(col_arr[col_arr >= 0])
        total_num += leng
        acc_dict[model_full_name] = (L, total_num)

    return above_threshold_image_dict, missing_object_image_dict, acc_dict, cleaned_df_col