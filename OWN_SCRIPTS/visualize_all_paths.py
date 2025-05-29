import random
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R

from mpc import ModelPredictiveControl

np.set_printoptions(precision=2, suppress=True)

# Path to the dataset folder
folder = "."

# Plot the trajectory with arrows
fig = plt.figure(figsize=(8, 6))



def change_transformation_matrix(old_pose, y_change, yaw_change):
    translation_change = np.array([0.0,y_change, 0.0])
    rotation_matrix = old_pose[:3, :3]
    old_translation = old_pose[:3, 3]

    # Calculate RPY angles using scipy.spatial.transform.Rotation
    old_yaw, pitch, roll = R.from_matrix(rotation_matrix).as_euler('zyx')  # Using ZYX Euler angles
    #old_x, old_y, old_z = old_translation

    new_yaw = old_yaw + yaw_change
    new_rotation = R.from_euler('zyx', [new_yaw, pitch, roll]).as_matrix()
    #new_translation = np.array([old_x, old_y + y_change, old_z])
    new_translation = old_translation + rotation_matrix @ translation_change

    new_pose = np.eye(4)  # Create an identity matrix
    new_pose[:3, :3] = new_rotation
    new_pose[:3, 3] = new_translation

    return new_pose, new_yaw, new_translation




def run_mpc_step(x, y, yaw, v, ego_poses, other_vehicle_locations=None):
    MPC = ModelPredictiveControl()
    MPC.set_initial_position(x, y, yaw, v)
    if other_vehicle_locations:
        MPC.set_obstacles(np.array(other_vehicle_locations))
    MPC.set_reference_trajectory_matrix(ego_poses)
    MPC.set_range(0, 1)
    MPC.set_lookahead_distance(150)
    MPC.set_target_velocity(v)

    steerings, throttles, waypoints_per_frame = MPC.simulate(plotting=False, rt_plotting=False)
    waypoints = waypoints_per_frame[0]
    steer = -steerings[1]
    throttle = throttles[1]
    return steer, throttle, waypoints







#############################
# EXTRACT EGO POSES
############################

# Lists to store trajectory data
x_vals, y_vals, yaw_angles = [], [], []
ego_car_poses = []

# Iterate over all frames
for i in range(0, 198):
    car_pose = np.loadtxt(f"{folder}/ego_pose/{i:06d}.txt")
    ego_car_poses.append(car_pose)
    # Extract car position
    car_pose_pos = car_pose[:3, 3]  # The translation part of the transformation matrix
    x_vals.append(car_pose_pos[0])
    y_vals.append(car_pose_pos[1])

    # Extract yaw angle (rotation around Z-axis)
    car_pose_rot = car_pose[:3, :3]  # Extract the 3x3 rotation matrix
    print("i", i)
    # print(car_pose_rot)
    yaw, pitch, roll = R.from_matrix(car_pose_rot).as_euler('zyx')  # Using ZYX Euler angles
    yaw_angles.append(-yaw)




#############################
# READ TRACK INFO TO PLOT OTHER CARS
############################

input_file = f"{folder}/track/track_info.txt"
data_all = pd.read_csv(input_file, sep='\s+', header=0)
data_other_vehicles = defaultdict(dict)


for _, row in data_all.iterrows():
    frame_id, track_id = int(row["frame_id"]), row['track_id']
    if frame_id >= 198:
        continue
    old_position_homogeneous = np.array([row['box_center_x'], row['box_center_y'], row['box_center_z'], 1])
    print(frame_id)
    car_pose_global = ego_car_poses[frame_id] @ old_position_homogeneous
    data_other_vehicles[track_id][frame_id] = car_pose_global[:2]





#############################
# Generate MPC trajectories
############################

num_random_mpc_trajectories = 15

mpc_path_plots = []
mpc_trajectories = []
mpc_start_idx = []
for trajectory_idx in range(num_random_mpc_trajectories):
    start_point_idx = random.randint(50,150)
    yaw_change = random.uniform(-0.7,0.8)
    y_change = random.uniform(1,5) * random.choice([-1,1])
    new_pose, new_yaw, new_translation = change_transformation_matrix(ego_car_poses[start_point_idx],
                                                                      y_change, yaw_change
                                                                      )
    plt.arrow(
        new_translation[0], new_translation[1],  # starting point (x, y)
        3 * np.cos(new_yaw), 3* np.sin(new_yaw),  # direction (dx, dy)
        color='red', head_width=0.7, head_length=1.2, length_includes_head=True
    )

    _,_, waypoints = run_mpc_step(x=new_translation[0], y=new_translation[1], yaw=new_yaw, v=8, ego_poses=ego_car_poses, other_vehicle_locations=None)
    mpc_trajectories.append(waypoints)
    mpc_start_idx.append(start_point_idx)

    # plot path
    plt.plot(waypoints[:,0], waypoints[:,1], color="black")

    # plot (moveable) point
    mpc_path_plots.append(plt.scatter(waypoints[0,0], waypoints[0,1], label=f"mpc{trajectory_idx}", color="yellow", zorder=3))






#############################
# PLOTTING
############################


# # PLOT OTHER VEHICLES AS LINES
# for k,v in data_other_vehicles.items():
#     if len(v) > 50:
#         print(len(v))
#         v = np.array(v)
#         plt.plot(v[:,0], v[:,1], label=k, color="gray")
#


# # PLOT OTHER VEHICLES AS MOVEABLE POINTS
car_plots = {}
for k, v in data_other_vehicles.items():
    if len(v.items()) > 30:
        start_frame_id = 0
        if start_frame_id in v.keys():
            car_plots[k] = plt.scatter(v[start_frame_id][0], v[start_frame_id][1], label=k, color="purple", zorder=3)


# Plot EGO VEHICLE AS MOVABLE POINTS
ego_car_plot = plt.scatter(x_vals[0], y_vals[0], label="ego", color="orange",zorder=3.2)



# Define arrow spacing
N = 10  # Adjust this value to control arrow density
arrow_length = 1.2

for i in range(0, len(x_vals), N):
    plt.arrow(x_vals[i], y_vals[i], 
              arrow_length * np.cos(-yaw_angles[i]), 
              arrow_length * np.sin(-yaw_angles[i]), 
              head_width=0.4, head_length=0.4, fc='r', ec='r')
    
    # Add numbering near the arrows
    plt.text(x_vals[i], y_vals[i], str(i), fontsize=6, color='black', ha='right')

# Draw the trajectory line
plt.plot(x_vals, y_vals, linestyle='-', color='b', alpha=0.6, label="Car Trajectory")

# Labels and formatting
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.title("Vehicle Trajectory with Yaw Direction and Numbered Arrows")
#plt.legend()
plt.grid(True)
plt.axis("equal")  # Keep aspect ratio correct


def frame_update(frame):
    frame = int(frame)

    # update otehr cars
    for k, v in data_other_vehicles.items():
        if k in car_plots and frame in v.keys():
            car_plots[k].set_offsets([[v[frame][0], v[frame][1]]])

    # update ego car
    ego_car_plot.set_offsets([[x_vals[frame], y_vals[frame]]])

    # update mpc_paths
    for trajectory_idx, mpc_trajectory in enumerate(mpc_trajectories):
        start_offset = mpc_start_idx[trajectory_idx]
        if 0 < frame - start_offset < len(mpc_trajectory):
            mpc_path_plots[trajectory_idx].set_offsets([[ mpc_trajectory[frame - start_offset][0], mpc_trajectory[frame - start_offset][1] ]])
    fig.canvas.draw_idle()


ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])  # [left, bottom, width, height]
slider = Slider(ax_slider, 'Frame', 0, 198, valinit=0, valstep=1)
slider.on_changed(frame_update)

# Show the plot
plt.show()

