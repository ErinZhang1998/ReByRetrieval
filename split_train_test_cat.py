import json 
from sklearn.model_selection import train_test_split
import numpy as np 

json_file_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/object_instances.json'
shape_categories_file_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/taxonomy_tabletop_small_keys.txt'

tabletop_small_training_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/tabletop_small_training_instances.json'
tabletop_small_testing_path = '/media/xiaoyuz1/hdd5/xiaoyuz1/data/tabletop_small_testing_instances.json'

shapenet_models = json.load(open(json_file_path))

table_top_categories = []
with open(shape_categories_file_path) as shape_categories_file:
    for line in shape_categories_file:
        if line.strip() == '':
            continue
        table_top_categories.append(line.strip())

shapenet_models_train = {}
shapenet_models_test = {}

for cat_name in table_top_categories:
    X_train, X_test  = train_test_split(shapenet_models[cat_name], test_size=0.3)
    shapenet_models_train[cat_name] = X_train
    shapenet_models_test[cat_name] = X_test


with open(tabletop_small_training_path, 'w+') as fp:
    json.dump(shapenet_models_train, fp)

with open(tabletop_small_testing_path, 'w+') as fp:
    json.dump(shapenet_models_test, fp)

shapenet_models_train = json.load(open(tabletop_small_training_path))
shapenet_models_test = json.load(open(tabletop_small_testing_path))