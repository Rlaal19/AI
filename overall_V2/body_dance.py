import cv2
import numpy as np
import mediapipe as mp
import math
import pygame

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize pygame mixer
pygame.mixer.init()
sounds = {
    "left": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_Dead Kick_Press_0115.6.wav'),
    "right": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_HiHat_Foot_1144.4.wav'),
    "head": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Overhead Sample 4.wav'),
    "chest": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Snare Sample 27.wav'),
}

# Initialize cameras
camera_indices = [0, 1, 2]  # Indices for up to 3 cameras
caps = [cv2.VideoCapture(idx) for idx in camera_indices if cv2.VideoCapture(idx).isOpened()]
if not caps:
    print("Error: No cameras available.")
    exit()

for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate the dynamic hitbox radius based on distance between shoulders
def calculate_dynamic_radius(shoulder_distance):
    base_distance = 0.4  # Base distance between shoulders in normalized coordinates
    base_radius = 100    # Base radius in pixels
    scaling_factor = shoulder_distance / base_distance
    return int(base_radius * scaling_factor)

# Function to check if the activation pose is detected
def is_activation_pose_detected(results):
    if not results.pose_landmarks:
        return False

    left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

    return (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)

fingertip_indices = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

program_active = False
current_hitbox = None
finger_inside_hitbox = False

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while True:
        frames = []
        overlays = []
        
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                continue
            frame = cv2.flip(frame, 1)
            frames.append(frame)

        if not frames:
            break

        combined_frame = np.zeros_like(frames[0])
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb_frame)
            hand_results = hands.process(rgb_frame)
            overlay = frame.copy()

            if is_activation_pose_detected(results):
                program_active = not program_active
                print("Program activated!" if program_active else "Program deactivated!")
                cv2.waitKey(500)

            if program_active:
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                    left_shoulder_x = int(left_shoulder.x * frame.shape[1])
                    left_shoulder_y = int((left_shoulder.y - 0.1) * frame.shape[0])
                    right_shoulder_x = int(right_shoulder.x * frame.shape[1])
                    right_shoulder_y = int((right_shoulder.y - 0.1) * frame.shape[0])

                    shoulder_distance = calculate_distance(
                        (left_shoulder.x, left_shoulder.y),
                        (right_shoulder.x, right_shoulder.y)
                    )
                    radius = calculate_dynamic_radius(shoulder_distance)
                    alpha = 0.4

                    head = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                    head_x = int(head.x * frame.shape[1])
                    head_y = int((head.y - 0.2) * frame.shape[0])
                    chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                    chest_y = int((left_shoulder.y + right_shoulder.y + 0.3) / 2 * frame.shape[0])

                    left_color = (0, 255, 0)
                    right_color = (0, 255, 0)
                    head_color = (0, 255, 0)
                    chest_color = (0, 255, 0)

                    new_hitbox = None
                    finger_detected = False

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            for idx in fingertip_indices:
                                fingertip = hand_landmarks.landmark[idx]
                                fingertip_x = int(fingertip.x * frame.shape[1])
                                fingertip_y = int(fingertip.y * frame.shape[0])

                                if calculate_distance((fingertip_x, fingertip_y), (left_shoulder_x, left_shoulder_y)) < radius:
                                    left_color = (0, 0, 255)
                                    new_hitbox = "left"
                                    finger_detected = True
                                if calculate_distance((fingertip_x, fingertip_y), (right_shoulder_x, right_shoulder_y)) < radius:
                                    right_color = (0, 0, 255)
                                    new_hitbox = "right"
                                    finger_detected = True
                                if calculate_distance((fingertip_x, fingertip_y), (head_x, head_y)) < radius:
                                    head_color = (0, 0, 255)
                                    new_hitbox = "head"
                                    finger_detected = True
                                if calculate_distance((fingertip_x, fingertip_y), (chest_x, chest_y)) < radius:
                                    chest_color = (0, 0, 255)
                                    new_hitbox = "chest"
                                    finger_detected = True

                    if finger_detected:
                        if new_hitbox != current_hitbox or not finger_inside_hitbox:
                            if new_hitbox in sounds:
                                sounds[new_hitbox].play()
                        finger_inside_hitbox = True
                    else:
                        finger_inside_hitbox = False

                    current_hitbox = new_hitbox

                    cv2.circle(overlay, (left_shoulder_x, left_shoulder_y), radius, left_color, -1)
                    cv2.circle(overlay, (right_shoulder_x, right_shoulder_y), radius, right_color, -1)
                    cv2.circle(overlay, (head_x, head_y), radius, head_color, -1)
                    cv2.circle(overlay, (chest_x, chest_y), radius, chest_color, -1)

                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            activation_status = "Active" if program_active else "Inactive"
            status_color = (0, 255, 0) if program_active else (0, 0, 255)
            cv2.putText(frame, f"Program: {activation_status}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

            combined_frame = cv2.addWeighted(combined_frame, 0.5, frame, 0.5, 0)

        cv2.imshow('Pose-based Sound Player (Multi-Camera)', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
