Real-time yoga pose recognition and correction using MediaPipe and Random Forest.

### Setup
Use Python 3.11.  
Create and activate a virtual environment: python -m venv venv_yoga .\venv_yoga\Scripts\activate  
Install dependencies: pip install mediapipe opencv-python pandas scikit-learn pyttsx4  

### Ensure the following files exist in your folder:

live_detection.py, landmarks.py, calc_angles.py, recommendations.p  
models/4_poses.model  
csv_files/4_angles_poses_angles.csv  

### How to Run
Open a terminal in the project folder.  
Activate the virtual environment if not already active.  
Run the command: python live_detection.py  
Stand back so the webcam sees your full body.  
Press 'q' to exit the application.  


### How it works
MediaPipe: Extracts 33 body landmarks.  
Random Forest: Classifies the pose (Tree, Plank, etc.).  
Correction: Compares your joint angles to a reference CSV and gives voice feedback.  

### Poses Supported
Adho Mukha Svanasana (Downward Dog)  
Phalakasana (Plank)   
Utkata Konasana (Goddess)  
Vrikshasana (Tree)  
