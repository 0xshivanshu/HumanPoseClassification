import pandas as pd
import os
import calc_angles

# The repo stores landmarks as x, y, z, visibility for 33 points
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", 
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", 
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", 
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", 
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
    "right_heel", "left_foot_index", "right_foot_index"
]

column_headers = []
for name in landmark_names:
    column_headers.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_v"])
df = pd.read_csv('./csv_files/4_poses_data_pose.csv')
df.columns = column_headers + ["pose"]

print("Calculating angles using your custom rangles function...")
all_angles = []

for index, row in df.iterrows():
    lp_dict = {}
    # Converted row to a DataFrame-like series for the function
    pose_row_df = pd.DataFrame([row])
    angles = calc_angles.rangles(pose_row_df, lp_dict)
    all_angles.append(angles + [row["pose"]])

angle_cols = [
    "armpit_left", "armpit_right", "elbow_left", "elbow_right", 
    "hip_left", "hip_right", "knee_left", "knee_right", 
    "ankle_left", "ankle_right", "pose"
]
angles_df = pd.DataFrame(all_angles, columns=angle_cols)

# Grouped by pose to get the reference (Average) angles
final_reference = angles_df.groupby("pose").mean().reset_index()
output_path = './csv_files/4_angles_poses_angles.csv'
final_reference.to_csv(output_path, index=False)

print(f"Success! Created {output_path}")