# Script for converting an .obj 3D model into 2048 colored point cloud as a .ply file
# Originally, it was supposed to be done "on the fly", when data set is imported,
# but it is too computationally expensive.
# Combine delete_undesired() with main() {models can be checked before converting them to PLY}

import pymeshlab as pml
import os
from tqdm import tqdm
import json


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


if __name__ == '__main__':
    main()
