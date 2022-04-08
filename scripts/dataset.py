# Script for deleting 3d models that belong to undesirable categories i.e. people, places

import pandas as pd
import os

models_dir = './resources/3d_models/'
csv_file = pd.read_csv('resources/metadata.csv')
csv_file = pd.DataFrame(csv_file, columns=['fullId', 'category', 'wnsynset', 'wnlemmas'])
del_cat = ['room', 'court', 'courtyard', 'person', 'homo,man,human being,human', ]

for item in del_cat:
    del_data = csv_file.loc[csv_file['wnlemmas'] == item]

    for data in del_data['fullId']:
        data = data.split(sep='.')[1]
        path = os.path.join(models_dir, data)
        try:
            os.remove(path + '.obj')
            os.remove(path + '.mtl')
        except FileNotFoundError:
            print(f'[WARNING] File {data} doesn\'t exist')


