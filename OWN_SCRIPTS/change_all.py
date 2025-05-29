import os
import shutil
import json
import random
import types
from functools import partial

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
    parser.add_argument("--folder", type=str, default="../data/waymo/training/carla_0012", help="Path to training folder")
    parser.add_argument("--run_mpc", action="store_true", help="Run MPC Controller")
    parser.add_argument("--skip_plot", action="store_true", help="Skip plotting")

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
MOVING = True  # whether or not the other cars should move

# Load the JSON file
with open(f"{cfg.folder}/timestamps_original.json", "r") as file:
    data = json.load(file)

if MOVING:
    # Get the FRAME data
    frame_data = data["FRAME"]

    # Replace all other entries with FRAME's data
    for key in data.keys():
        if key != "FRAME":
            data[key] = frame_data

else:  # STATIC OTHER CARS
    # Get the timestamp of FRAME 150
    i_fixed = cfg.start_frame + 50
    frame_150_timestamp = data["FRAME"][f"{i_fixed:06d}"]

    # Replace all other entries with FRAME 150's timestamp
    for key1, value1 in data.items():
        for key2 in data[key1].keys():
            new_timestamp = int(key2) * 0.0001 + frame_150_timestamp
            data[key1][key2] = new_timestamp

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

poses_original = []
poses_new = {}

modes = ["constant_offset", "random", "sinus"]
mode = "sinus"

# for x, y, yaw, index in zip(x_list, y_list, yaw_list, range(cfg.start_frame, cfg.end_frame)):
for index in range(cfg.start_frame, cfg.end_frame):
    # i_fixed = cfg.start_frame + 50

    old_ego_pose = np.loadtxt(os.path.join(cfg.folder, "ego_pose_original", f"{index:06d}.txt"))
    poses_original.append(old_ego_pose)
    # IF LEFT TO RIGHT OR ROTATE
        # i_fixed = cfg.start_frame + 50
        # old_ego_pose = np.loadtxt(os.path.join(cfg.folder, "ego_pose_original", f"{index:06d}.txt"))
        # y_change = 0.0 # ((index - cfg.start_frame)/100.0 * 12) - 6 # for left-to-right
        # yaw_change = ((index - cfg.start_frame)/100.0 * 2 * 3.1415) - 3.1415  # for left-to-right
    yaw_change = 0.0
    y_change = 0.0

    if mode == "left_to_right":
        y_change = ((index - cfg.start_frame) / 100.0 * 12) - 6

    if mode == "random":
        y_change = random.uniform(*cfg.random_translation)
        yaw_change = random.uniform(*cfg.random_rotation)

    if mode == "sinus":
        y_change = np.sin(index * np.pi / 20) * 2

    if mode == "constant_offset":
        y_change = 5.0

    new_ego_pose = change_transformation_matrix(old_ego_pose, y_change, yaw_change)
    poses_new[index] = new_ego_pose
    y_changes[index] = y_change
    yaw_changes[index] = yaw_change
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

    #data_all.loc[data_all["frame_id"] == index, ['track_id', 'object_class', 'alpha', 'box_height', 'box_width', 'box_length','box_center_x', 'box_center_y', 'box_center_z', 'box_heading', 'speed']] \
    #    = data[['track_id', 'object_class', 'alpha', 'box_height', 'box_width', 'box_length','box_center_x', 'box_center_y', 'box_center_z', 'box_heading', 'speed']]

    # data_all.fillna(-1).astype({'track_id': int})
data_all.to_csv(output_track_file, sep=' ', index=False, header=True)

print("Step 3: Modified ego poses ran SUCCESSFULLY")
print("Step 4: Modified track_info.txt (poses of other cars) ran SUCCESSFULLY")





################################################################
#  STEP 5: Run MPC
################################################################
# MPC = False
#
# mpc_steering_angles = {}
# mpc_acceleration = {}
#
#
# from mpc_functions import *
#
# def do_simulation_step(cx, cy, cyaw, ck, sp, dl, initial_state):
#     """
#     Simulation
#
#     cx: course x position list
#     cy: course y position list
#     cy: course yaw position list
#     ck: course curvature list
#     sp: speed profile
#     dl: course tick [m]
#
#     """
#
#     # goal = [cx[-1], cy[-1]]
#
#     state = initial_state
#
#     # initial yaw compensation
#     if state.yaw - cyaw[0] >= math.pi:
#         state.yaw -= math.pi * 2.0
#     elif state.yaw - cyaw[0] <= -math.pi:
#         state.yaw += math.pi * 2.0
#
#     x = [state.x]
#     y = [state.y]
#     target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
#
#     odelta, oa = None, None
#
#     cyaw = smooth_yaw(cyaw)
#
#     # while MAX_TIME >= time:
#
#     sim_length = 20 if show_animation else 1
#     for i in range(sim_length):
#         xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)
#
#         time = i
#
#         x0 = [state.x, state.y, state.v, state.yaw]  # current state
#
#         oa, odelta, ox, oy, poseoyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)
#
#         if odelta is None:
#             return state.x, state.y, state.yaw, state.v, None, None
#
#         di, ai = 0.0, 0.0
#         if odelta is not None:
#             di, ai = odelta[0], oa[0]
#             steering_string = f"{'left' if di > 0 else 'right '} {abs(di):.4f}"
#             print("steering", steering_string, "acceleration", ai)
#             state = update_state(state, ai, di)
#
#
#         if show_animation:  # pragma: no cover
#
#             plt.cla()
#             # for stopping simulation with the esc key.
#             plt.gcf().canvas.mpl_connect('key_release_event',
#                     lambda event: [exit(0) if event.key == 'escape' else None])
#             if ox is not None:
#                 plt.plot(ox, oy, "xr", label="MPC")
#             plt.plot(cx, cy, "-r", label="course")
#
#             for i, (x, y) in enumerate(zip(cx, cy)):
#                 plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom')
#
#             plt.plot(x, y, "ob", label="trajectory")
#             plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
#             plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
#             plot_car(state.x, state.y, state.yaw, steer=di)
#             plt.axis("equal")
#             plt.legend()
#             plt.grid(True)
#             plt.title("Time[s]:" + str(round(time, 2))
#                       + ", speed[km/h]:" + str(round(state.v * 3.6, 2))
#                       + ", STEERING:" + steering_string)
#             #plt.pause(2)
#             #plt.show()
#             plt.pause(0.01)
#     if show_animation:
#         print("NEXT")
#     return state.x, state.y, state.yaw, state.v, di, ai
#
#
# def get_waymo_course(poses, dl):
#     ax = [pose[0,3] for pose in poses]
#     ay = [pose[1,3] for pose in poses]
#     cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
#
#     # cyaw = [i - math.pi for i in cyaw]
#     cyaw = [i for i in cyaw]
#
#     return cx, cy, cyaw, ck
#
# def transformation_matrix_to_x_y_yaw(pose):
#     pos_x = pose[0, 3]
#     pos_y = pose[1, 3]
#
#     dir_x = pose[0, 0]
#     dir_y = pose[1, 0]
#     yaw  = math.atan2(dir_y, dir_x)#  + math.pi
#     return pos_x, pos_y, yaw
#
#
#
# if MPC:
#
#     dl = 5.0  # course tick
#     cx, cy, cyaw, ck = get_waymo_course(poses_original, dl)
#
#     for index in range(cfg.start_frame, cfg.end_frame):
#         sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
#         start_x, start_y, start_yaw = transformation_matrix_to_x_y_yaw(poses_new[index])
#         initial_state = State(x=start_x, y=start_y, yaw=start_yaw, v=5.0)
#         show_animation = False
#         x, y, yaw, v, di, ai = do_simulation_step(cx, cy, cyaw, ck, sp, dl, initial_state)
#         mpc_steering_angles[index] = di
#         mpc_acceleration[index] = ai

mpc_steering_angles = {}
if cfg.run_mpc:

    # get Trajectory as (N,4) array
    trajectory_x = [pose[0,3] for pose in poses_original]
    trajectory_y = [pose[1,3] for pose in poses_original]
    trajectory_phi = [R.from_matrix(pose[:3, :3]).as_euler('zyx')[0] for pose in poses_original]

    # get initial pose as (4,) array
    init_pose_matrix = poses_original[0]
    yaw, _, _ = R.from_matrix(init_pose_matrix[:3, :3]).as_euler('zyx')  # Using ZYX Euler angle
    initial_pose = np.array([init_pose_matrix[0,3], init_pose_matrix[1,3], yaw, 0.0])

    from mpc import ModelPredictiveControl
    mpc = ModelPredictiveControl()

    mpc.set_initial_position(initial_pose)
    mpc.set_reference_trajectory(np.stack([trajectory_x, trajectory_y, trajectory_phi], axis=1))
    mpc.set_lookahead_distance(8.0)
    mpc.set_range(cfg.start_frame, cfg.end_frame)

    # read other vehicles
    input_file = f"{cfg.folder}/track/track_info.txt"
    data_all = pd.read_csv(input_file, sep='\s+', header=0)

    def run_at_step_i(frame_id, data_all, poses_new):
        data = data_all[data_all["frame_id"] == frame_id]
        obstacles = []
        for _, row in data.iterrows():
            old_position_homogeneous = [row['box_center_x'], row['box_center_y'], row['box_center_z'], 1]
            global_position = poses_new[frame_id] @ old_position_homogeneous
            obstacles.append(np.array([global_position[0], global_position[1], 1]))

        # set obstacles
        mpc.set_obstacles(np.array(obstacles))
    mpc.simulate(partial(run_at_step_i, data_all=data_all, poses_new=poses_new))




    data_info = {
        "y_changes": y_changes,
        "yaw_changes": yaw_changes,
        "mpc_steering_angles": mpc_steering_angles,
    }
    # Save to a JSON file
    with open(f"{cfg.folder}/info_data.json", "w") as json_file:
        json.dump(data_info, json_file, indent=4)  # `indent=4` makes it pretty-printed


    print("Step 5: Generate MPC steering angles ran SUCCESSFULLY")







################################################################
#  STEP 6: Plot
################################################################
if not cfg.skip_plot:
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
        ax.set_xlim(min(all_positions[:, 0]) - 10, max(all_positions[:, 0]) + 10)
        ax.set_ylim(min(all_positions[:, 1]) - 10, max(all_positions[:, 1]) + 10)
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
        start_yaw = np.arctan2(start_dir_y, start_dir_x)
        ax.arrow(start_x, start_y, start_dir_x * 2, start_dir_y * 2, head_width=0.5, head_length=1.4, fc='black', ec="blue")


        ### READ TRACK INFO TO PLOT OTHER CARS
        data_all = pd.read_csv(input_file, sep='\s+', header=0)
        data = data_all[data_all["frame_id"] == INDEX + cfg.start_frame]

        for _, row in data.iterrows():
            old_position_homogeneous = [row['box_center_x'], row['box_center_y'], row['box_center_z'], 1]
            global_position_car = pose_new @ old_position_homogeneous
            global_yaw_car = np.deg2rad(row['box_heading']) + start_yaw
            global_dir_car = np.array([np.cos(global_yaw_car),  np.sin(global_yaw_car)])

            # ax.scatter(global_position[0], global_position[1])
            ax.arrow(global_position_car[0], global_position_car[1], global_dir_car[0], global_dir_car[1], head_width=0.5, head_length=1.4)

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