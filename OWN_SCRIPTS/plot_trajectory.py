import os
import random
import math
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


np.set_printoptions(formatter={'float_kind': '{:0.3f}'.format})

# List of 4x4 transformation matrices
poses_original = []
poses_modified = []
input_folder = "ego_pose"
input_folder_orig = "ego_pose_original"


def transformation_matrix_to_pos_and_dir(pose):
    pos_x = pose[0, 3]
    pos_y = pose[1, 3]

    dir_x = pose[0, 0]
    dir_y = pose[1, 0]
    return pos_x, pos_y, dir_x, dir_y


def make_transformation_matrix(old_pose, new_x, new_y, new_yaw):
    rotation_matrix = old_pose[:3, :3]
    translation = old_pose  [:3, 3]

    # Calculate RPY angles using scipy.spatial.transform.Rotation
    _, pitch, roll = R.from_matrix(rotation_matrix).as_euler('zyx')  # Using ZYX Euler angles
    old_z = translation[2]

    new_rotation = R.from_euler('zyx', [new_yaw, pitch, roll]).as_matrix()
    new_translation = np.array([new_x, new_y, old_z])

    new_pose = np.eye(4)  # Create an identity matrix
    new_pose[:3, :3] = new_rotation
    new_pose[:3, 3] = new_translation

    return new_pose


# Define transformation function to map coordinates and heading to a new frame
def transform_coordinates_and_heading(old_position, old_heading, old_pose, new_pose):
    """
    Transform a 3D point and heading from old_pose to new_pose.
    Args:
        old_position (ndarray): [x, y, z] coordinates in the old frame.
        old_heading (float): Heading angle in the old frame.
        old_pose (ndarray): 4x4 matrix representing the old pose.
        new_pose (ndarray): 4x4 matrix representing the new pose.
    Returns:
        tuple: Transformed coordinates and heading in the new frame.
    """
    # Convert position to homogeneous coordinates
    old_position_homogeneous = np.array([*old_position, 1.0], dtype=np.float32)

    # Compute the inverse of the old pose
    inverse_old_pose = np.linalg.inv(old_pose)

    # Transform to the global frame
    global_position = np.dot(inverse_old_pose, old_position_homogeneous)

    # Transform to the new frame
    new_position_homogeneous = np.dot(new_pose, global_position)

    # Extract x, y, z coordinates
    print("ZZZZ", new_position_homogeneous[2])
    new_position = new_position_homogeneous[:3]

    # Compute the heading transformation
    old_rotation = np.arctan2(old_pose[1, 0], old_pose[0, 0])
    new_rotation = np.arctan2(new_pose[1, 0], new_pose[0, 0])
    heading_offset = new_rotation - old_rotation
    new_heading = old_heading + heading_offset

    return new_position, new_heading


# Iterate over all files in the input folder
for i, filename in enumerate(sorted(os.listdir(input_folder))):
    if filename.endswith('.txt') and not "_" in filename:
        if 98 < int(filename[:-4]) < 198:
            input_file_path = os.path.join(input_folder, filename)
            poses_modified.append(np.loadtxt(input_file_path))

# Iterate over all files in the input folder
for i, filename in enumerate(sorted(os.listdir(input_folder_orig))):
    if filename.endswith('.txt') and not "_" in filename:
        if 98 < int(filename[:-4]) < 198:
            input_file_path = os.path.join(input_folder_orig, filename)
            poses_original.append(np.loadtxt(input_file_path))

fig, ax = plt.subplots()

for matrices, color in zip([poses_original, poses_modified], ['green', 'red']):
    # Extract positions (x, y) from the fourth column of each matrix
    positions = np.array([[m[0, 3], m[1, 3]] for m in matrices])

    # Extract heading direction from the third column
    directions = np.array([[m[0, 0], m[1, 0]] for m in matrices])

    # Plot with arrows
    for pos, head in zip(positions, directions):
        start_x, start_y = pos
        dir_x, dir_y = head
        ax.arrow(start_x, start_y, dir_x * 0.5, dir_y * 0.5,
                 head_width=0.2, head_length=0.4, fc='black', ec=color)

# Enhancements for visibility
all_positions = np.array([[m[0, 3], m[1, 3]] for m in poses_original + poses_modified])
ax.set_xlim(min(all_positions[:, 0]) - 2, max(all_positions[:, 0]) + 2)
ax.set_ylim(min(all_positions[:, 1]) - 2, max(all_positions[:, 1]) + 2)
ax.set_aspect('equal')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Trajectory')
plt.grid(True)
#plt.show()

####################################################################

start_matrix = poses_modified[0]  # 4x4 np array
goal_pos = poses_original[7] # 4x4 array

start_x, start_y, start_dir_x, start_dir_y = transformation_matrix_to_pos_and_dir(start_matrix)
goal_x, goal_y, goal_dir_x, goal_dir_y = transformation_matrix_to_pos_and_dir(goal_pos)

ax.arrow(start_x, start_y, start_dir_x * 0.5, start_dir_y * 0.5,
         head_width=0.2, head_length=0.4, fc='black', ec="blue")
ax.arrow(goal_x, goal_y, goal_dir_x * 0.5, goal_dir_y * 0.5,
         head_width=0.2, head_length=0.4, fc='black', ec="blue")

plt.show()


