# Scripts for sampling a 3D mesh with UV colour information

import pymeshlab as pml

mesh = pml.MeshSet()
mesh.load_new_mesh('resources/3d_models/001/model_fixed.obj')
mesh.generate_sampling_montecarlo(samplenum=2048)
mesh.transfer_texture_to_color_per_vertex(sourcemesh=0, targetmesh=1)
mesh.save_current_mesh('resources/3d_models/001/model_fixed.ply')

