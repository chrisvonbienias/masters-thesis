import pymeshlab as pml
import os
from tqdm import tqdm

def main() -> None:

    partial_path = 'resources/depth_images/pcd'
    partial_save_path = 'resources/dataset/partial'
    complete_path = 'resources/dataset/complete'
    categories = os.listdir(partial_path)

    for cat in tqdm(categories, desc='Processing categories', leave=False):
        models = os.listdir(os.path.join(partial_path, cat))
        for model in tqdm(models):
            mesh = pml.MeshSet()  # type: ignore
            mesh.set_verbosity(False)
            mesh.load_new_mesh(os.path.join(complete_path, model + '.ply'))
            mesh.load_new_mesh(os.path.join(partial_path, cat, model, '0.ply'))
            mesh.compute_matrix_from_rotation(rotaxis='X axis', rotcenter='origin', angle=180)
            mesh.transfer_attributes_per_vertex(sourcemesh=0, targetmesh=1)
            ply_path = os.path.join(partial_save_path, model + '.ply')
            mesh.save_current_mesh(ply_path)

if __name__ == '__main__':
    main()