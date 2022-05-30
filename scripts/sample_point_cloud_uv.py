# Script for converting an .obj 3D model into 16384 colored point cloud as a .ply file
# Originally, it was supposed to be done "on the fly", when dataset is imported,
# but it is too computationally expensive.

import pymeshlab as pml
import os
from tqdm import tqdm


def main() -> None:
    
    models_path = 'resources/3d_models'
    categories = os.listdir(models_path)
    save_path = 'resources/dataset'

    for category in tqdm(categories[46:]):
        cat_path = f'{models_path}/{category}'
        models = os.listdir(cat_path)

        for model in tqdm(models, leave=False):
            mesh_path = f'{cat_path}/{model}/models/model_normalized.obj'
            mesh = pml.MeshSet()
            try:
                mesh.load_new_mesh(mesh_path)
            except:
                continue
            mesh.compute_color_transfer_face_to_vertex()
            mesh.generate_sampling_montecarlo(samplenum=16384, radiusvariance=0.5, exactnum=True)
            mesh.transfer_attributes_per_vertex(sourcemesh=0, targetmesh=1)
            ply_path = f'{save_path}/{model}.ply'
            mesh.save_current_mesh(ply_path)

    print('Models converted')


if __name__ == '__main__':
    main()
