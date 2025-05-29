import numpy as np

mat = np.loadtxt("../data/waymo/training/002/extrinsics/1.txt")
rot = mat[:3,:3]
trans = mat[:3, 3]

print("R", rot, "\nt", trans)
