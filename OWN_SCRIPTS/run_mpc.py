import os
import random
import math
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from mpc_functions import State, angle_mod, pi_2_pi, get_linear_model_matrix, plot_car, update_state, get_nparray_from_matrix, calc_nearest_index, predict_motion, iterative_linear_mpc_control, linear_mpc_control, calc_ref_trajectory, check_goal, smooth_yaw, calc_speed_profile, get_switch_back_course
from mpc_functions import DT, MAX_TIME, TARGET_SPEED, show_animation
from mpc_functions import cubic_spline_planner


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
def transform_coordinates_and_heading(old_position_homogeneous, old_heading, old_pose, new_pose):
    """
    Transform a 3D point and heading from old_pose to new_pose.
    Args:
        old_position_homogeneous (ndarray): [x, y, z] coordinates in the old frame.
        old_heading (float): Heading angle in the old frame.
        old_pose (ndarray): 4x4 matrix representing the old pose.
        new_pose (ndarray): 4x4 matrix representing the new pose.
    Returns:
        tuple: Transformed coordinates and heading in the new frame.
    """
    # Convert position to homogeneous coordinates
    # old_position_homogeneous = np.array([*old_position, 1.0], dtype=np.float32)

    # # Compute the inverse of the old pose
    # inverse_old_pose = np.linalg.inv(old_pose)
    # # Transform to the global frame
    # global_position = np.dot(inverse_old_pose, old_position_homogeneous)
    # # Transform to the new frame
    # new_position_homogeneous = np.dot(new_pose, global_position)

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


def make_poses(x_list, y_list, yaw_list):
    input_file = 'track_original/track_info.txt'
    output_file = 'track/track_info.txt'
    data_all = pd.read_csv(input_file, sep='\s+', header=0)

    for x, y, yaw, index in zip(x_list, y_list, yaw_list, range(98, 199)):
        for suffix in [".txt", "_0.txt", "_1.txt", "_2.txt", "_3.txt", "_4.txt"]:
            old_ego_pose = np.loadtxt(os.path.join("ego_pose_original", f"{index:06d}" + suffix))
            new_ego_pose = make_transformation_matrix(old_ego_pose, x, y, new_yaw=yaw + 3.14)
            output_file_path = os.path.join("ego_pose", f"{index:06d}{suffix}")
            assert new_ego_pose.shape == (4,4)
            np.savetxt(output_file_path, new_ego_pose, fmt='%.18e')
            # np.savetxt(output_file_path, new_ego_pose, fmt='%.2f')

            if suffix == ".txt":
                data = data_all[data_all["frame_id"] == index]

                #transformed_data = [
                #    transform_coordinates_and_heading(
                #        [row['box_center_x'], row['box_center_y'], row['box_center_z']],
                #        row['box_heading'], old_ego_pose, new_ego_pose #new_ego_pose2
                #    ) for _, row in data.iterrows()
                #]
                #
                #positions, headings = zip(*transformed_data) if transformed_data else ([], [])

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
                data_all.loc[
                    data_all["frame_id"] == index, ['box_center_x', 'box_center_y', 'box_center_z', 'box_heading']] = \
                data[['box_center_x', 'box_center_y', 'box_center_z', 'box_heading']]

    data_all.to_csv(output_file, sep=' ', index=False, header=True)
    print(f"Modified file saved as {output_file}")


def random_y():
    number = random.uniform(3,6)
    sign = random.choice([-1, 1])
    return number * sign


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)

        time = time + DT

        dist_to_trajectory = math.sqrt((state.x - cx[target_ind])**2 + (state.y - cy[target_ind])**2)
        if dist_to_trajectory < 0.6:
            state.y += random_y()

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        print("len", len(x))
        if len(x) == 100:
            make_poses(x,y,yaw)
            raise RuntimeError("DONE")


        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a


def get_waymo_course(poses, dl):
    ax = [pose[0,3] for pose in poses]
    ay = [pose[1,3] for pose in poses]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck



def main3():
    # get trajectory
    input_folder_orig = "ego_pose_original"
    poses_original = []
    # Iterate over all files in the input folder
    for i, filename in enumerate(sorted(os.listdir(input_folder_orig))):
        if filename.endswith('.txt') and not "_" in filename:
            if 98 < int(filename[:-4]) < 198:
                input_file_path = os.path.join(input_folder_orig, filename)
                poses_original.append(np.loadtxt(input_file_path))

    print(__file__ + " start!!")
    start = time.time()

    dl = 5.0  # course tick
    cx, cy, cyaw, ck = get_waymo_course(poses_original, dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    start_x, start_y, start_dir_x, start_dir_y = transformation_matrix_to_pos_and_dir(poses_original[0])

    initial_state = State(x=start_x, y=start_y + 6, yaw=math.atan2(start_dir_y, start_dir_x) + 3.14, v=-4.0)

    t, x, y, yaw, v, d, a = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()

main3()