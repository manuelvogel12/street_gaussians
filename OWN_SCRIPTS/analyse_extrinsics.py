import numpy as np
from scipy.spatial.transform import Rotation as R

# Function to read a transformation matrix from a file and compute translation and rotation
def extract_transformation_and_euler_angles(file_path):
    # Load the matrix from the file using numpy's loadtxt function
    matrix = np.loadtxt(file_path)
    
    # Extract translation (x, y, z) from the last column
    translation = matrix[:3, 3]
    
    # Extract rotation matrix (top-left 3x3 block)
    rotation_matrix = matrix[:3, :3]
    
    # Use scipy to compute Euler angles (roll, pitch, yaw) from the rotation matrix
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('zyx', degrees=False)  # 'zyx' order: yaw-pitch-roll
    
    return translation, euler_angles

# Example usage with file '0.txt'
file_path = '/home/manuel/ma/street_gaussians/data/waymo/training/002/extrinsics/2.txt'
translation, euler_angles = extract_transformation_and_euler_angles(file_path)

# Print results
print(f"Translation (x, y, z): {translation}")
print(f"Rotation (roll, pitch, yaw) [in radians]: {euler_angles}")
