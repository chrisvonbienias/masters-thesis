import os
import json
from sklearn.model_selection import train_test_split
import numpy as np

models_path = 'resources/3d_models'
categories = os.listdir(models_path)
cat_dict = {}

f = open('resources/model_list.txt', 'w')

train_count = 0
test_count = 0
valid_count = 0

for category in categories:
    cat_path = f'{models_path}/{category}'
    models = os.listdir(cat_path)
    train, test = train_test_split(models, test_size=0.3, random_state=42)
    validate, test = train_test_split(test, test_size=0.3, random_state=42)
    train = [[model, 'train'] for model in train]
    test = [[model, 'test'] for model in test]
    validate = [[model, 'validate'] for model in validate]
    train_count += len(train)
    test_count += len(test)
    valid_count += len(validate)
    cat_dict[category] = [*train, *test, *validate]

    f.writelines(f'{category}/{model}\n' for model in models)

f.close()

with open('resources/categories.json', 'w') as f:
    json.dump(cat_dict, f)

print(f'Train: {train_count}')
print(f'Test: {test_count}')
print(f'Validate: {valid_count}')
print(f'Total: {train_count+test_count+valid_count}')
