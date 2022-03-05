# Scripts for sampling a 3D mesh with UV colour information

import trimesh
import numpy as np
from matplotlib import pyplot as plt

trimesh.util.attach_to_log()

# Point sampling of 3D mesh.
# A modified version of trimesh.sample with UV color sampling
# Returns an array n*6 [x, y, z, r, g, b]
def sample_with_uv(mesh):
    pass


mesh = trimesh.load_mesh('resources/3d_models/001/model_fixed.obj')

points = mesh.sample(2048)
points_uv = sample_with_uv(mesh)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()
