# Scripts for sampling a 3D mesh with UV colour information

import trimesh
import numpy as np
from matplotlib import pyplot as plt

trimesh.util.attach_to_log()


def sample_with_uv(mesh):
    pass


mesh = trimesh.load_mesh('dataset/model_fixed.obj')

points = mesh.sample(2048)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()
