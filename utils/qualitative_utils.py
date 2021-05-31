import numpy as np 

def indices_to_cat_and_id(dataset, sample_ids):
    obj_cat = []
    obj_id = []
    for sample_id in list(sample_ids):
        idx = dataset.sample_id_to_idx[sample_id]
        sample = dataset.idx_to_data_dict[idx]
        obj_cat.append(sample['obj_cat'])
        obj_id.append(sample['obj_id'])
    
    return np.array(obj_cat), np.array(obj_id)
