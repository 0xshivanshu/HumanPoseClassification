import os
import cv2
import mediapipe as mp
import pandas as pd

dataset_path = r"C:\Users\thesh\.cache\kagglehub\datasets\niharika41298\yoga-poses-dataset\versions\1\DATASET\TRAIN"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
data = []

print(f"Scanning images in: {dataset_path}")

if not os.path.exists(dataset_path):
    print("Error: Path not found. Check if 'DATASET/TRAIN' exists in the cache folder.")
else:
    for pose_label in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, pose_label)
        if os.path.isdir(label_folder):
            print(f"Processing pose: {pose_label}")
            for img_name in os.listdir(label_folder):
                img = cv2.imread(os.path.join(label_folder, img_name))
                if img is None: continue
                
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                    landmarks.append(pose_label)
                    data.append(landmarks)

    df = pd.DataFrame(data)
    df.to_csv('final_data.csv', index=False)
    print("Success! final_data.csv created.")