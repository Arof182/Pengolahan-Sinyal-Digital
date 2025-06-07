import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_resp(frame, start_time):
    # Konversi BGR ke RGB tanpa flip horizontal
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        y_nose = landmarks[0].y
        y_chest = landmarks[1].y
        resp = y_nose - y_chest
    else:
        resp = 0

    t = time.time() - start_time
    return resp, t
