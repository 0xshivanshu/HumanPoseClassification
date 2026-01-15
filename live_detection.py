import cv2
from time import time
import pickle as pk
import mediapipe as mp
import pandas as pd
import pyttsx4
import multiprocessing as mtp

from recommendations import check_pose_angle
from landmarks import extract_landmarks
from calc_angles import rangles


def init_cam():
    cam = cv2.VideoCapture(0) 
    if not cam.isOpened():
        print("Could not open webcam index 0, trying index 1...")
        cam = cv2.VideoCapture(1)
    return cam


def get_pose_name(index):
    names = {
        0: "Adho Mukha Svanasana",
        1: "Phalakasana",
        2: "Utkata Konasana",
        3: "Vrikshasana",
    }
    return str(names[index])


def init_dicts():
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", 
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", 
        "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", 
        "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
        "right_heel", "left_foot_index", "right_foot_index"
    ]
    col_names = []
    for name in landmark_names:
        col_names.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_v"])
    
    landmarks_points_array = {
        "left_shoulder": [], "right_shoulder": [],
        "left_elbow": [], "right_elbow": [],
        "left_wrist": [], "right_wrist": [],
        "left_hip": [], "right_hip": [],
        "left_knee": [], "right_knee": [],
        "left_ankle": [], "right_ankle": [],
        "left_heel": [], "right_heel": [],
        "left_foot_index": [], "right_foot_index": [],
    }
    
    return col_names, landmarks_points_array

# engine = pyttsx4.init() 

def tts(tts_q):
    import pyttsx4
    engine = pyttsx4.init()
    while True:
        objects = tts_q.get()
        if objects is None:
            break
        message = objects[0]
        engine.say(message)
        engine.runAndWait()
    tts_q.task_done()


def cv2_put_text(image, message):
    cv2.putText(
        image,
        message,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        5,
        cv2.LINE_AA
    )


def destory(cam, tts_proc, tts_q):
    cv2.destroyAllWindows()
    cam.release()
    tts_q.put(None)
    tts_q.close()
    tts_q.join_thread()
    tts_proc.join()


if __name__ == "__main__":
    print("1. Initializing Camera")
    cam = init_cam()
    
    print("2. Loading AI Model")
    model = pk.load(open("./models/4_poses.model", "rb"))
    cols, landmarks_points_array = init_dicts()
    print("3. Loading Reference CSV ")
    angles_df = pd.read_csv("./csv_files/4_angles_poses_angles.csv")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    print("4. Starting Voice Background Process")
    tts_q = mtp.JoinableQueue()

    tts_proc = mtp.Process(target=tts, args=(tts_q, ))
    tts_proc.start()
    print("5. Main Loop Starting (Press 'q' in the window to quit)")
    tts_last_exec = time() + 5

    while True:
        result, image = cam.read()
        if result:
            flipped = cv2.flip(image, 1)
        
            resized_image = cv2.resize(
                flipped,
                (640, 360),
                interpolation=cv2.INTER_AREA
            )

            err, df, landmarks = extract_landmarks(
                resized_image,
                mp_pose,
                cols
            )

            if err == False:
                prediction = model.predict(df.values)
                probabilities = model.predict_proba(df.values)
                label_map = {
                    "Adho Mukha Svanasana": 0,
                    "Phalakasana": 1,
                    "Utkata Konasana": 2,
                    "Vrikshasana": 3,
                }

                pred_label = prediction[0]
                pred_idx = label_map.get(pred_label, 0)

                mp_drawing.draw_landmarks(
                    flipped,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                if probabilities[0, pred_idx] > 0.50:
                    cv2_put_text(
                        flipped,
                        pred_label
                    )

                    angles = rangles(df, landmarks_points_array)
                    suggestions = check_pose_angle(
                        pred_idx, angles, angles_df)

                    if time() > tts_last_exec:
                        tts_q.put([suggestions[0]])
                        tts_last_exec = time() + 5
                else:
                    cv2_put_text(flipped, "No Pose Detected")
            
            cv2.imshow("Frame", flipped)

        key = cv2.waitKey(1)
        if key == ord("q"):
            destory(cam, tts_proc, tts_q)
            break