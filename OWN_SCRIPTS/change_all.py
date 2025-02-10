import os
import shutil
import json
import random
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse

################################################################
#  STEP 0: Parameters
################################################################


def get_config():
    parser = argparse.ArgumentParser(description="Waymo dataset preprocessing configuration")

    parser.add_argument("--start_frame", type=int, default=97, help="Starting frame index")
    parser.add_argument("--end_frame", type=int, default=197, help="Ending frame index")
    parser.add_argument("--random_translation", type=float, nargs=2, default=[-6.0, 6.0], help="Random translation range")
    parser.add_argument("--random_rotation", type=float, nargs=2, default=[-0.4, 0.4], help="Random rotation range")
    parser.add_argument("--folder", type=str, default="../data/waymo/training/0000", help="Path to training folder")

    return parser.parse_args()

cfg = get_config()





################################################################
#  STEP 1: Backup files
################################################################

def rename_if_not_exist(original_name):
    backup_name = f"{original_name}_original"
    if os.path.exists(original_name) and not os.path.exists(backup_name):
        shutil.move(original_name, backup_name)
        os.makedirs(original_name, exist_ok=False)

# Rename folders if their original versions do not exist
rename_if_not_exist(f"{cfg.folder}/ego_pose")
rename_if_not_exist(f"{cfg.folder}/track")

# Rename file timestamps.json if timestamps_original.json does not exist
if os.path.exists(f"{cfg.folder}/timestamps.json") and not os.path.exists(f"{cfg.folder}/timestamps_original.json"):
    shutil.move(f"{cfg.folder}/timestamps.json", f"{cfg.folder}/timestamps_original.json")


print("Step 1: Move directories (if necessary) ran SUCCESSFULLY")







################################################################
#  STEP 2: Change Timestamps
################################################################

# Load the JSON file
with open(f"{cfg.folder}/timestamps_original.json", "r") as file:
    data = json.load(file)

# Get the FRAME data
frame_data = data["FRAME"]

# Replace all other entries with FRAME's data
for key in data.keys():
    if key != "FRAME":
        data[key] = frame_data

# Save the updated JSON file
with open(f"{cfg.folder}/timestamps.json", "w") as file:
    json.dump(data, file, indent=2)

print("Step 2: Updated timestamps.json ran SUCCESSFULLY")






################################################################
#  STEP 3,4: Update Trajectory &  Other car's positions
################################################################

def change_transformation_matrix(old_pose, y_change, yaw_change):
    translation_change = np.array([0.0,y_change, 0.0])
    rotation_matrix = old_pose[:3, :3]
    old_translation = old_pose[:3, 3]

    # Calculate RPY angles using scipy.spatial.transform.Rotation
    old_yaw, pitch, roll = R.from_matrix(rotation_matrix).as_euler('zyx')  # Using ZYX Euler angles
    #old_x, old_y, old_z = old_translation

    new_rotation = R.from_euler('zyx', [old_yaw + yaw_change, pitch, roll]).as_matrix()
    #new_translation = np.array([old_x, old_y + y_change, old_z])
    new_translation = old_translation + rotation_matrix @ translation_change

    new_pose = np.eye(4)  # Create an identity matrix
    new_pose[:3, :3] = new_rotation
    new_pose[:3, 3] = new_translation

    return new_pose


# Define transformation function to map coordinates and heading to a new frame
def transform_coordinates_and_heading(old_position_homogeneous, old_heading, old_pose, new_pose):
    """
    Transform a 3D point and heading from old_pose to new_pose.
    Args:
        old_position_homogeneous (ndarray): [x, y, z, 1] coordinates in the old frame.
        old_heading (float): Heading angle in the old frame.
        old_pose (ndarray): 4x4 matrix representing the old pose.
        new_pose (ndarray): 4x4 matrix representing the new pose.
    Returns:
        tuple: Transformed coordinates and heading in the new frame.
    """

    # Transform to the global frame
    global_position = np.dot(old_pose, old_position_homogeneous)

    # Transform to the new frame
    inverse_new_pose = np.linalg.inv(new_pose)
    new_position = np.dot(inverse_new_pose, global_position)[:3]


    # Compute the heading transformation
    old_rotation = np.arctan2(old_pose[1, 0], old_pose[0, 0])
    new_rotation = np.arctan2(new_pose[1, 0], new_pose[0, 0])
    heading_offset = new_rotation - old_rotation
    new_heading = old_heading - heading_offset

    return new_position, new_heading



# Copy files
shutil.copytree(f"{cfg.folder}/ego_pose_original", f"{cfg.folder}/ego_pose", dirs_exist_ok=True)
shutil.copytree(f"{cfg.folder}/track_original", f"{cfg.folder}/track", dirs_exist_ok=True)

input_track_file = f"{cfg.folder}/track_original/track_info.txt"
output_track_file = f"{cfg.folder}/track/track_info.txt"


### READ TRACK INFO
data_all = pd.read_csv(input_track_file, sep='\s+', header=0)

y_changes = {}
yaw_changes = {}
mpc_steering_angles = {}

# for x, y, yaw, index in zip(x_list, y_list, yaw_list, range(cfg.start_frame, cfg.end_frame)):
for index in range(cfg.start_frame, cfg.end_frame):
    old_ego_pose = np.loadtxt(os.path.join(cfg.folder, "ego_pose_original", f"{index:06d}.txt"))
    y_change = random.uniform(*cfg.random_translation)
    yaw_change = random.uniform(*cfg.random_rotation)
    new_ego_pose = change_transformation_matrix(old_ego_pose, y_change, yaw_change)
    y_changes[index] = y_change
    yaw_changes[index] = yaw_change
    mpc_steering_angles[index] = 0.0 # TODO
    assert new_ego_pose.shape == (4, 4)
    for suffix in [".txt", "_0.txt", "_1.txt", "_2.txt", "_3.txt", "_4.txt"]:
        output_file_path = os.path.join(cfg.folder, "ego_pose", f"{index:06d}{suffix}")
        np.savetxt(output_file_path, new_ego_pose, fmt='%.18e')

    data = data_all[data_all["frame_id"] == index]
    positions, headings = [], []
    for _, row in data.iterrows():
        old_position = np.array([row['box_center_x'], row['box_center_y'], row['box_center_z'], 1], dtype=np.float32)
        position, heading = transform_coordinates_and_heading(old_position, row['box_heading'], old_ego_pose, new_ego_pose)
        positions.append(position)
        headings.append(heading)

    if positions:
        positions = np.array(positions)
        data.loc[:, ['box_center_x', 'box_center_y', 'box_center_z']] = positions
        data.loc[:, 'box_heading'] = headings

    # Update data_all with the modified data
    data_all.loc[data_all["frame_id"] == index, ['box_center_x', 'box_center_y', 'box_center_z', 'box_heading']] \
        = data[['box_center_x', 'box_center_y', 'box_center_z', 'box_heading']]

data_all.to_csv(output_track_file, sep=' ', index=False, header=True)

data_info = {
    "y_changes": y_changes,
    "yaw_changes": yaw_changes,
    "mpc_steering_angles": mpc_steering_angles,
}

# Save to a JSON file
with open(f"{cfg.folder}/info_data.json", "w") as json_file:
    json.dump(data_info, json_file, indent=4)  # `indent=4` makes it pretty-printed


print("Step 3: Modified ego poses ran SUCCESSFULLY")
print("Step 4: Modified track_info.txt (poses of other cars) ran SUCCESSFULLY")







################################################################
#  STEP 5: Plot
################################################################
input_folder = f"{cfg.folder}/ego_pose"
input_folder_orig = f"{cfg.folder}/ego_pose_original"
poses_modified = []
poses_original = []


def transformation_matrix_to_pos_and_dir(pose):
    pos_x, pos_y  = pose[0:2, 3]
    dir_x, dir_y = pose[0:2, 0]
    return pos_x, pos_y, dir_x, dir_y


# Iterate over all files in the modified input folder
for i, filename in enumerate(sorted(os.listdir(input_folder))):
    if filename.endswith('.txt') and not "_" in filename:
        if (cfg.start_frame <= int(filename[:-4]) < cfg.end_frame):
            input_file_path = os.path.join(input_folder, filename)
            poses_modified.append(np.loadtxt(input_file_path))

# Iterate over all files in the original input folder
for i, filename in enumerate(sorted(os.listdir(input_folder_orig))):
    if filename.endswith('.txt') and not "_" in filename:
        if cfg.start_frame <= int(filename[:-4]) < cfg.end_frame:
            input_file_path = os.path.join(input_folder_orig, filename)
            poses_original.append(np.loadtxt(input_file_path))

fig, ax = plt.subplots()
for INDEX in range(cfg.end_frame - cfg.start_frame): # e.g. 0..100
    print(INDEX)

    ### PLOT THE TRAJECTORIES (GREEN + RED)
    for matrices, color in zip([poses_original, poses_modified], ['green', 'red']):
        #                           # pos_x     pos_y     dir_x   dir_y
        positions_and_directions = [[m[0, 3], m[1, 3], m[0, 0], m[1, 0]] for m in matrices]
        # Plot with arrows
        for start_x, start_y, dir_x, dir_y in positions_and_directions:
            ax.arrow(start_x, start_y, dir_x * 0.5, dir_y * 0.5, head_width=0.2, head_length=0.4, fc='black', ec=color)

    # Enhancements for visibility
    all_positions = np.array([[m[0, 3], m[1, 3]] for m in poses_original + poses_modified])
    ax.set_xlim(min(all_positions[:, 0]) - 40, max(all_positions[:, 0]) + 40)
    ax.set_ylim(min(all_positions[:, 1]) - 40, max(all_positions[:, 1]) + 40)
    ax.set_aspect('equal')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Trajectory')
    plt.grid(True)

    #####################################################
    input_file = f"{cfg.folder}/track/track_info.txt"

    pose_new = poses_modified[INDEX]  # 4x4 np array
    pose_old = poses_original[INDEX]  # 4x4 np array
    assert pose_new.shape == (4,4)
    assert pose_old.shape == (4,4)


    ### PLOT CURRENT CAR POSE IN BLUE
    start_x, start_y, start_dir_x, start_dir_y = transformation_matrix_to_pos_and_dir(pose_new)
    ax.arrow(start_x, start_y, start_dir_x * 2, start_dir_y * 2, head_width=0.5, head_length=1.4, fc='black', ec="blue")


    ### READ TRACK INFO TO PLOT OTHER CARS
    data_all = pd.read_csv(input_file, sep='\s+', header=0)
    data = data_all[data_all["frame_id"] == INDEX + cfg.start_frame]

    for _, row in data.iterrows():
        old_position_homogeneous = [row['box_center_x'], row['box_center_y'], row['box_center_z'], 1]
        global_position = pose_new @ old_position_homogeneous
        ax.scatter(global_position[0], global_position[1])

    ### SAVE MIDDLE FRAME
    if INDEX == 50:
        plt.savefig(f"{cfg.folder}/overview.png")

    ### SHOW PLOT
    if os.getenv("PYCHARM_HOSTED"):
        plt.show()
        fig, ax = plt.subplots()
    else:
        plt.pause(1)
        ax.clear()










########## OLD CODE #############################################################


################################################################
#  STEP 3: Update Trajectory
################################################################
#
# # Input and output folder paths
# input_folder = 'ego_pose_original'
# output_folder = 'ego_pose'
#
#
# # OFFSET_X = 0.0
# # OFFSET_Y = 0.0
# # OFFSET_Z = 0.0
#
# # # Iterate over all files in the input folder
# # for i, filename in enumerate(sorted(os.listdir(input_folder))):
# #     if filename.endswith('.txt'):
# #         input_file_path = os.path.join(input_folder, filename)
# #         output_file_path = os.path.join(output_folder, filename)
# #
# #         # Load data from the input file
# #         data = np.loadtxt(input_file_path)
# #         if i > cfg.start_frame * 5:
# #             # pos_x
# #             data[0, 3] += OFFSET_X
# #
# #             # pos_y
# #             data[1, 3] += OFFSET_Y
# #
# #             # pos_z
# #             data[2, 3] += OFFSET_Z
# #
# #         # Save the modified data to the output file
# #         np.savetxt(output_file_path, data, fmt='%.18e')
#
# for i in range(cfg.start_frame, cfg.end_frame):
#     input_file_path = os.path.join(input_folder, f"{i:06}.txt")
#
#     data = np.loadtxt(input_file_path)
#     data[1, 3] += random.randrange(*cfg.random_translation) # random y offset
#     for suffix in ["", "_0", "_1", "_2", "_3", "_4"]:
#         output_file_path = os.path.join(output_folder, f"{i:06}{suffix}.txt")
#         np.savetxt(output_file_path, data, fmt='%.18e')
#
#
# print("Step 3: Modified ego poses ran SUCCESSFULLY")
#
# ################################################################
# #  STEP 4: Update Other car's positions
# ################################################################
#
# # Define the input and output file paths
# input_file = 'track_original/track_info.txt'
# output_file = 'track/track_info.txt'
#
#
# ### READ TRACK INFO
# data_all = pd.read_csv(input_file, sep='\s+', header=0)
#
# ####################################################################################
#
# for INDEX in range(cfg.start_frame, cfg.end_frame):
#     data = data_all[data_all["frame_id"] == INDEX]
#
#     data['box_center_x'] = pd.to_numeric(data['box_center_x'], errors='coerce')
#     data['box_center_y'] = pd.to_numeric(data['box_center_y'], errors='coerce')
#     data['box_center_z'] = pd.to_numeric(data['box_center_z'], errors='coerce')
#
#     #data['box_center_x'] = data['box_center_x'] - OFFSET_X
#     data['box_center_y'] = data['box_center_y'] - (random.randrange(*cfg.random_translation) / 2)
#     #data['box_center_z'] = data['box_center_z'] - (OFFSET_Z / 200)
#
#     # Save the modified data back to a file
#     data.to_csv(output_file, sep=' ', index=False, header=False)
#
# print(f"Modified file saved as {output_file}")