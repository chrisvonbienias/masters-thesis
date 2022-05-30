import os
import json

models_path = 'resources/3d_models'
categories = os.listdir(models_path)
cat_dict = {}

for category in categories:
    cat_path = f'{models_path}/{category}'
    models = os.listdir(cat_path)
    cat_dict[category] = models

with open('resources/categories.json', 'w') as f:
    json.dump(cat_dict, f)

