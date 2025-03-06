import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=2, suppress=True)

# Path to the dataset folder
folder = "../data/waymo/training/carla_0008"

# Lists to store trajectory data
x_vals, y_vals, yaw_angles = [], [], []

# Iterate over all frames
for i in range(0, 198):
    car_pose = np.loadtxt(f"{folder}/ego_pose/{i:06d}.txt")
    
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

# Plot the trajectory with arrows
plt.figure(figsize=(8, 6))

# Define arrow spacing
N = 10  # Adjust this value to control arrow density
arrow_length = 1.2  # Make arrows longer

for i in range(0, len(x_vals), N):
    plt.arrow(x_vals[i], y_vals[i], 
              arrow_length * np.cos(-yaw_angles[i]), 
              arrow_length * np.sin(-yaw_angles[i]), 
              head_width=0.4, head_length=0.4, fc='r', ec='r')
    
    # Add numbering near the arrows
    plt.text(x_vals[i], y_vals[i], str(i), fontsize=10, color='black', ha='right')

# Draw the trajectory line
plt.plot(x_vals, y_vals, linestyle='-', color='b', alpha=0.6, label="Car Trajectory")

# Labels and formatting
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.title("Vehicle Trajectory with Yaw Direction and Numbered Arrows")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Keep aspect ratio correct

# Show the plot
plt.show()

