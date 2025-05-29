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



# Define transformation function to map coordinates and heading to a new frame
def transform_coordinates_and_heading(old_position, old_heading, old_pose, new_pose, ax):
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
    # inverse_old_pose = np.linalg.inv(old_pose)
    inverse_new_pose = np.linalg.inv(new_pose)

    # Transform to the global frame
    if FROM_ORIGINAL_POSES:
        global_position = np.dot(old_pose, old_position_homogeneous)
    else:
        global_position = np.dot(new_pose, old_position_homogeneous)
    ax.scatter(global_position[0], global_position[1])
    # global_position = np.dot(inverse_old_pose, old_position_homogeneous)

    # Transform to the new frame
    # new_position_homogeneous = np.dot(new_pose, global_position)
    new_position_homogeneous = np.dot(inverse_new_pose, global_position)

    # Extract x, y, z coordinates
    new_position = new_position_homogeneous[:3]

    # Compute the heading transformation
    old_rotation = np.arctan2(old_pose[1, 0], old_pose[0, 0])
    new_rotation = np.arctan2(new_pose[1, 0], new_pose[0, 0])
    heading_offset = new_rotation - old_rotation
    new_heading = old_heading + heading_offset

    return new_position, new_heading


def plot_pose(ax, pos, direction, **kwargs):
    start_x, start_y = pos[:2]
    dir_x, dir_y = direction
    ax.arrow(start_x, start_y, dir_x * 0.5, dir_y * 0.5, **kwargs)


# Iterate over all files in the modified input folder
for i, filename in enumerate(sorted(os.listdir(input_folder))):
    if filename.endswith('.txt') and not "_" in filename:
        if (98 <= int(filename[:-4]) < 198):
            input_file_path = os.path.join(input_folder, filename)
            poses_modified.append(np.loadtxt(input_file_path))

# Iterate over all files in the original input folder
for i, filename in enumerate(sorted(os.listdir(input_folder_orig))):
    if filename.endswith('.txt') and not "_" in filename:
        if 98 <= int(filename[:-4]) < 198:
            input_file_path = os.path.join(input_folder_orig, filename)
            poses_original.append(np.loadtxt(input_file_path))


####################################################################
fig, ax = plt.subplots()
for INDEX in range(100):
    print(INDEX)
    ax.clear()    

    for matrices, color in zip([poses_original, poses_modified], ['green', 'red']):
        # Extract positions (x, y) from the fourth column of each matrix
        positions = np.array([[m[0, 3], m[1, 3]] for m in matrices])

        # Extract heading direction from the third column
        directions = np.array([[m[0, 0], m[1, 0]] for m in matrices])

        # Plot with arrows
        for pos, direction in zip(positions, directions):
            plot_pose(ax, pos, direction, head_width=0.2, head_length=0.4, fc='black', ec=color)

    # Enhancements for visibility
    all_positions = np.array([[m[0, 3], m[1, 3]] for m in poses_original + poses_modified])
    ax.set_xlim(min(all_positions[:, 0]) - 2, max(all_positions[:, 0]) + 2)
    ax.set_ylim(min(all_positions[:, 1]) - 40, max(all_positions[:, 1]) + 40)
    ax.set_aspect('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Trajectory')
    plt.grid(True)
    # plt.show()

    #####################################################
    FROM_ORIGINAL_POSES = False
    if FROM_ORIGINAL_POSES:
        input_file = 'track_original/track_info.txt'
    else:  # use modified poses
        input_file = 'track/track_info.txt'


    start_pose_new = poses_modified[INDEX]  # 4x4 np array
    start_pose_old = poses_original[INDEX]  # 4x4 np array
    start_pose = start_pose_old if FROM_ORIGINAL_POSES else start_pose_new
    assert start_pose_new.shape == (4,4)
    assert start_pose_old.shape == (4,4)

    start_x, start_y, start_dir_x, start_dir_y = transformation_matrix_to_pos_and_dir(start_pose)

    ax.arrow(start_x, start_y, start_dir_x * 2, start_dir_y * 2,
             head_width=0.5, head_length=1.4, fc='black', ec="blue")


    ### READ TRACK INFO
    data_all = pd.read_csv(input_file, sep='\s+', header=0)
    data = data_all[data_all["frame_id"] == INDEX + 98]


    transformed_data = [
        transform_coordinates_and_heading(
            [row['box_center_x'], row['box_center_y'], row['box_center_z']],
            row['box_heading'], start_pose_old, start_pose_new, ax
        ) for _, row in data.iterrows()
    ]
    positions, headings = zip(*transformed_data) if transformed_data else ([], [])
    for pos, heading_angle in zip(positions, headings):
        heading_vector = [np.sin(heading_angle), np.cos(heading_angle)]
        plot_pose(ax, pos, heading_vector, head_width=0.2, head_length=0.4, fc='black', ec="purple")
    # if transformed_data:
    #     positions = np.array(positions)
    #     data.loc[:, ['box_center_x', 'box_center_y', 'box_center_z']] = positions
    #     data.loc[:, 'box_heading'] = headings
    #
    # # Update data_all with the modified data
    # data_all.loc[
    #     data_all["frame_id"] == index, ['box_center_x', 'box_center_y', 'box_center_z', 'box_heading']] = \
    # data[['box_center_x', 'box_center_y', 'box_center_z', 'box_heading']]
    #
    # data_all.to_csv(output_file, sep=' ', index=False, header=True)
    # print(f"Modified file saved as {output_file}")

    plt.pause(0.1)





