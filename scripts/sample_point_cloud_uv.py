# Script for converting an .obj 3D model into 2048 colored point cloud as a .ply file
# Originally, it was supposed to be done "on the fly", when data set is imported,
# but it is too computationally expensive.
# Combine delete_undesired() with main() {models can be checked before converting them to PLY}

import pymeshlab as pml
import os
from tqdm import tqdm
import pandas as pd


# Script for deleting models belonging to undesirable classes
def delete_undesired():
    models_dir = './resources/3d_models/'
    csv_file = pd.read_csv('resources/metadata.csv')
    csv_file = pd.DataFrame(csv_file, columns=['fullId', 'category', 'wnsynset', 'wnlemmas'])
    del_cat = ['room', 'court', 'courtyard', 'person', 'homo,man,human being,human', ]

    deleted = 0
    for item in tqdm(del_cat, desc='Processing models...'):
        del_data = csv_file.loc[csv_file['wnlemmas'] == item]

        for data in del_data['fullId']:
            data = data.split(sep='.')[1]
            path = os.path.join(models_dir, data)
            try:
                os.remove(path + '.obj')
                os.remove(path + '.mtl')
                os.remove(path + '.ply')
                deleted += 1
            except FileNotFoundError:
                print(f'[WARNING] File {data} doesn\'t exist')

    print("Operation finished")
    print(f"Deleted {deleted} models")


def main() -> None:
    models_path = 'resources/3d_models/'
    models = os.listdir(models_path)
    # remove .mtl and .jpg files
    models = [x for x in models if x[-1] == 'j']

    removed_models = []
    for model in tqdm(models, desc='Processing models...', position=0, leave=False):
        mesh_path = models_path + model
        mesh = pml.MeshSet()
        mesh.load_new_mesh(mesh_path)

        if mesh.current_mesh().texture_number():  # if mesh has a texture
            mesh.generate_sampling_montecarlo(samplenum=2048)
            mesh.transfer_texture_to_color_per_vertex(sourcemesh=0, targetmesh=1)
            ply_path = mesh_path.split(sep='.')[0] + '.ply'
            mesh.save_current_mesh(ply_path)
            # print(f'Saving {ply_path}')
        else:
            removed_models.append(mesh)
            os.remove(models_path + model)
            mtl_path = model.split('.')[0]
            os.remove(models_path + mtl_path + '.mtl')
            # print(f'[WARNING] Model {model} has no texture')

    print('Models converted')
    print(f'Removed {len(removed_models)} models')

    delete_undesired()


if __name__ == '__main__':
    main()
