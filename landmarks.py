import pandas as pd
import numpy as np
import cv2

def extract_landmarks(image, mp_pose, cols):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # MediaPipe expects RGB, but OpenCV uses BGR
        result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if result.pose_landmarks:
            all_list = []
            for landmark in result.pose_landmarks.landmark:
                all_list.extend([
                    landmark.x, 
                    landmark.y, 
                    landmark.z, 
                    landmark.visibility
                ])
            
            return False, pd.DataFrame([all_list], columns=cols), result.pose_landmarks
    
        return True, pd.DataFrame(), None