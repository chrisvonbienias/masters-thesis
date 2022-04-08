# Script for sampling 3D meshes in ShapeNet with UV colour information
# Sampled meshes are exported to corresponding folders as PLY

import os
import pymeshlab as pml

models_path = 'resources/3d_models/'
folders = os.listdir(models_path)
folders.pop(-1)  # removes '3d_models.txt'
'''
for folder in folders:
    mesh = pml.MeshSet()
    mesh.load_new_mesh(models_path + folder + 'model.obj')
    mesh.generate_sampling_montecarlo(samplenum=2048)
    mesh.transfer_texture_to_color_per_vertex(sourcemesh=0, targetmesh=1)
    mesh.save_current_mesh(models_path + folder + 'model.ply')
'''
