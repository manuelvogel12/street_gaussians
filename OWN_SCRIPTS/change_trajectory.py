import os
import numpy as np
import random
import pandas as pd


#################################
### CHANGE EGO VEHICLE POSES ###
################################

# Input and output folder paths
input_folder = 'ego_pose_original'
output_folder = 'ego_pose'

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)


OFFSET_X = 0.0
OFFSET_Y = 6.0
OFFSET_Z = 0.0

# Iterate over all files in the input folder
for i, filename in enumerate(sorted(os.listdir(input_folder))):
    #if i%50 == 0:
    #    OFFSET_Z = float(random.randint(-300,300))
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        # Load data from the input file
        data = np.loadtxt(input_file_path)
        if i > 98*5:
            # pos_x
            data[0, 3] += OFFSET_X
            
            # pos_y
            data[1, 3] += OFFSET_Y
            
            # pos_z
            data[2, 3] += OFFSET_Z

        # Save the modified data to the output file
        np.savetxt(output_file_path, data, fmt='%.18e')

print("Processing complete. Modified files are saved in the 'ego_pose' folder.")


####################################
###   CHANGE OTHER VEHICLE POSES ###
####################################

# Define the input and output file paths
input_file = 'track_original/track_info.txt'
output_file = 'track/track_info.txt'

# Load the data from the file using pandas
column_names = [
    'frame_id', 'track_id', 'object_class', 'alpha', 'box_height', 'box_width', 'box_length',
    'box_center_x', 'box_center_y', 'box_center_z', 'box_heading', 'speed'
]

# # Read the file with whitespace as separator
data = pd.read_csv(input_file, sep='\s+', names=column_names)
data['box_center_x'] = pd.to_numeric(data['box_center_x'], errors='coerce')
data['box_center_y'] = pd.to_numeric(data['box_center_y'], errors='coerce')
data['box_center_z'] = pd.to_numeric(data['box_center_z'], errors='coerce')

data['box_center_x'] = data['box_center_x'] - OFFSET_X
data['box_center_y'] = data['box_center_y'] - (OFFSET_Y / 2)
data['box_center_z'] = data['box_center_z'] - (OFFSET_Z/200)

# Save the modified data back to a file
data.to_csv(output_file, sep=' ', index=False, header=False)

print(f"Modified file saved as {output_file}")
