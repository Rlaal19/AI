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
    "left": pygame.mixer.Sound(r'/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_Dead Kick_Press_0115.6.wav'),
    "right": pygame.mixer.Sound(r'/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_HiHat_Foot_1144.4.wav'),
    "head": pygame.mixer.Sound(r'/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Overhead Sample 4.wav'),
    "chest": pygame.mixer.Sound(r'/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Snare Sample 27.wav'),
    "hip": pygame.mixer.Sound(r'/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Tom Sample 17.wav'),
}

# Initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate the dynamic hitbox radius based on distance between shoulders
def calculate_dynamic_radius(shoulder_distance):
    base_distance = 0.4  # Base distance between shoulders in normalized coordinates
    base_radius = 100    # Base radius in pixels
    scaling_factor = shoulder_distance / base_distance
    return int(base_radius * scaling_factor)

# Hand landmark indices for fingertips
fingertip_indices = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

# State trackers for playing sound
current_hitbox = None
last_hitbox = None
finger_inside_hitbox = False

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while True:
        # Capture each frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for a natural mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Holistic
        results = holistic.process(rgb_frame)
        
        # Process the frame with Mediapipe Hands (for finger detection)
        hand_results = hands.process(rgb_frame)

        # Overlay initialization
        overlay = frame.copy()

        # Draw landmarks and connections for pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Get the left shoulder position
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            left_shoulder_x = int(left_shoulder.x * frame.shape[1])
            left_shoulder_y = int((left_shoulder.y - 0.3) * frame.shape[0])

            # Get the right shoulder position
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            right_shoulder_x = int(right_shoulder.x * frame.shape[1])
            right_shoulder_y = int((right_shoulder.y - 0.3) * frame.shape[0])

            # Calculate the distance between shoulders in normalized coordinates
            shoulder_distance = calculate_distance(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y),
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
            )

            # Calculate dynamic radius for hitboxes
            radius = calculate_dynamic_radius(shoulder_distance)
            alpha = 0.4  # Transparency factor

            # Get the head position
            head = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            head_x = int(head.x * frame.shape[1])
            head_y = int((head.y - 0.5) * frame.shape[0])

            # Get the chest (breast) position
            chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
            chest_y = int((left_shoulder.y + right_shoulder.y + 0.3) / 2 * frame.shape[0])

            # Get the hip position and adjust Y-coordinate to move it higher
            left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
            hip_x = int((left_hip.x + right_hip.x) / 2 * frame.shape[1])
            hip_y = int(((left_hip.y + right_hip.y) / 2 - 0.1) * frame.shape[0])  # ลดค่า Y ลง 0.1 เพื่อยกตำแหน่งขึ้น

            # Default colors for hitboxes
            left_color = (0, 255, 0)  # Green
            right_color = (0, 255, 0)  # Green
            head_color = (0, 255, 0)  # Green
            chest_color = (0, 255, 0)  # Green
            hip_color = (0, 255, 0)  # Green

            new_hitbox = None
            finger_detected = False

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for idx in fingertip_indices:
                        fingertip = hand_landmarks.landmark[idx]
                        fingertip_x = int(fingertip.x * frame.shape[1])
                        fingertip_y = int(fingertip.y * frame.shape[0])

                        # Check if any finger touches the left shoulder circle
                        left_distance = calculate_distance((fingertip_x, fingertip_y), (left_shoulder_x, left_shoulder_y))
                        if left_distance < radius:
                            left_color = (0, 0, 255)  # Change to red when touched
                            new_hitbox = "left"
                            finger_detected = True

                        # Check if any finger touches the right shoulder circle
                        right_distance = calculate_distance((fingertip_x, fingertip_y), (right_shoulder_x, right_shoulder_y))
                        if right_distance < radius:
                            right_color = (0, 0, 255)  # Change to red when touched
                            new_hitbox = "right"
                            finger_detected = True

                        # Check if any finger touches the head circle
                        head_distance = calculate_distance((fingertip_x, fingertip_y), (head_x, head_y))
                        if head_distance < radius:
                            head_color = (0, 0, 255)  # Change to red when touched
                            new_hitbox = "head"
                            finger_detected = True

                        # Check if any finger touches the chest circle
                        chest_distance = calculate_distance((fingertip_x, fingertip_y), (chest_x, chest_y))
                        if chest_distance < radius:
                            chest_color = (0, 0, 255)  # Change to red when touched
                            new_hitbox = "chest"
                            finger_detected = True

                        # Check if any finger touches the hip circle
                        hip_distance = calculate_distance((fingertip_x, fingertip_y), (hip_x, hip_y))
                        if hip_distance < radius:
                            hip_color = (0, 0, 255)  # Change to red when touched
                            new_hitbox = "hip"
                            finger_detected = True

            # Play sound if the finger is detected and conditions are met
            if finger_detected:
                if new_hitbox != current_hitbox or not finger_inside_hitbox:
                    if new_hitbox in sounds:
                        sounds[new_hitbox].play()
                finger_inside_hitbox = True
            else:
                finger_inside_hitbox = False

            current_hitbox = new_hitbox

            # Draw translucent circles on shoulders, head, chest, and hip
            cv2.circle(overlay, (left_shoulder_x, left_shoulder_y), radius, left_color, -1)
            cv2.circle(overlay, (right_shoulder_x, right_shoulder_y), radius, right_color, -1)
            cv2.circle(overlay, (head_x, head_y), radius, head_color, -1)
            cv2.circle(overlay, (chest_x, chest_y), radius, chest_color, -1)
            cv2.circle(overlay, (hip_x, hip_y), radius, hip_color, -1)

            # Add labels to each hitbox
            cv2.putText(frame, "DI_Dead Kick_Press_Sound🥁", (left_shoulder_x - radius, left_shoulder_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "DI_HiHat_FootFoot_Sound🥁", (right_shoulder_x - radius, right_shoulder_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Overhead_Sound🥁", (head_x - radius, head_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Snare_Sound🥁", (chest_x - radius, chest_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Tom_Sound🥁", (hip_x - radius, hip_y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display the frame
        cv2.imshow('Shoulder, Head, Chest, and Hip Tracker with Finger Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
