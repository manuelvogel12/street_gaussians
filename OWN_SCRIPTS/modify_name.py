import json
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True, help="ID to specify the directory")
args = parser.parse_args()

ID = args.ID

# Define paths
json_path = f"data/waymo/training/{ID}/info_data.json"
output_dir = f"output/waymo_full_exp/waymo_train_{ID}/trajectory/training_data"
image_dir_base = f"output/waymo_full_exp/waymo_train_{ID}/trajectory"

image_dir = None
if os.path.exists(os.path.join(image_dir_base, "ours_50000")):
    image_dir = os.path.join(image_dir_base, "ours_50000")
elif os.path.exists(os.path.join(image_dir_base, "ours_20000")):
    image_dir = os.path.join(image_dir_base, "ours_20000")
elif os.path.exists(os.path.join(image_dir_base, "ours_10000")):
    image_dir = os.path.join(image_dir_base, "ours_10000")
else:
    raise RuntimeError("Directory does not exist")


os.makedirs(output_dir, exist_ok=False) # training_data already existent

# Load JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

y_changes = data["y_changes"]
yaw_changes = data["yaw_changes"]
mpc_steering_angles = data["mpc_steering_angles"]


for i in range(97, 197):
    y = y_changes.get(str(i))
    yaw = yaw_changes.get(str(i))
    yaw_deg = int(yaw * 180 / 3.1415)
    mpc_angle = mpc_steering_angles.get(str(i))
    if mpc_angle is None or abs(mpc_angle) > 0.989 :
        continue
    
    if y is None or yaw is None:
        raise RuntimeError(f"Missing data in JSON: {i}")
    
    old_filename = f"{image_dir}/{i:06d}_0_rgb.png"
    steering_string = f"{'left' if mpc_angle > 0 else 'right'}_{abs(mpc_angle):.3f}"

    new_filename = f"{output_dir}/front_{i:03d}_y_{y:.2f}_yaw_{yaw_deg}_mpc_{steering_string}.png"
    
    if os.path.exists(old_filename):
        shutil.copy(old_filename, new_filename)
        print(f"Copied: {old_filename} -> {new_filename}")
    else:
        raise RuntimeError(f"File not found: {old_filename}")

