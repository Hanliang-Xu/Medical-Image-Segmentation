from Project.surface import *
import numpy as np


# Creating gts and t1s instances
gts = surface()
t1s = surface()

gts.verts = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]  # Bottom
])

gts.faces = np.array([
    [0, 1, 2], [0, 2, 3],  # Bottom
])

t1s.verts = np.array([
    [400, 2, 1], [400, 4, 1], [-10000, 2, 1], [-10000, 4, 1]  # Bottom
])

t1s.faces = np.array([
    [0, 1, 2], [0, 2, 3],  # Bottom
])

MASDg1, HDg1, mp1, mp2 = gts.surfDistances(t1s)

print(MASDg1, HDg1)