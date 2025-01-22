import cv2
import numpy as np
import mediapipe as mp
import math
import pygame

# Initialize Mediapipe Holistic and Hands
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
    "hip": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Tom Sample 17.wav'),
}

# Initialize up to 3 cameras
camera_indices = [0, 1, 2]
caps = [cv2.VideoCapture(i) for i in camera_indices if cv2.VideoCapture(i).isOpened()]

# Check if cameras opened successfully
if not caps:
    print("Error: No cameras available.")
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

# State trackers for each player
players = [{} for _ in caps]

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    print("\nPress 'q' to quit.")
    while True:
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Camera {idx} failed to provide a frame.")
                continue

            # Flip the frame horizontally for a natural mirror effect
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with Mediapipe Holistic
            results = holistic.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            # Overlay initialization
            overlay = frame.copy()

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Get key body positions
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                head = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]

                # Calculate key coordinates and dynamic radius
                left_shoulder_x, left_shoulder_y = int(left_shoulder.x * frame.shape[1]), int((left_shoulder.y - 0.3) * frame.shape[0])
                right_shoulder_x, right_shoulder_y = int(right_shoulder.x * frame.shape[1]), int((right_shoulder.y - 0.3) * frame.shape[0])
                head_x, head_y = int(head.x * frame.shape[1]), int((head.y - 0.5) * frame.shape[0])
                chest_x, chest_y = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]), int((left_shoulder.y + right_shoulder.y + 0.3) / 2 * frame.shape[0])
                hip_x, hip_y = int((left_hip.x + right_hip.x) / 2 * frame.shape[1]), int(((left_hip.y + right_hip.y) / 2 - 0.1) * frame.shape[0])

                shoulder_distance = calculate_distance(
                    (left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y)
                )
                radius = calculate_dynamic_radius(shoulder_distance)

                # Default colors
                colors = {
                    "left": (0, 255, 0),
                    "right": (0, 255, 0),
                    "head": (0, 255, 0),
                    "chest": (0, 255, 0),
                    "hip": (0, 255, 0),
                }

                # Detect touch with hands
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        for idx in fingertip_indices:
                            fingertip = hand_landmarks.landmark[idx]
                            fingertip_x, fingertip_y = int(fingertip.x * frame.shape[1]), int(fingertip.y * frame.shape[0])

                            # Check collisions and update colors
                            for part, (x, y) in [("left", (left_shoulder_x, left_shoulder_y)),
                                                 ("right", (right_shoulder_x, right_shoulder_y)),
                                                 ("head", (head_x, head_y)),
                                                 ("chest", (chest_x, chest_y)),
                                                 ("hip", (hip_x, hip_y))]:
                                if calculate_distance((fingertip_x, fingertip_y), (x, y)) < radius:
                                    colors[part] = (0, 0, 255)
                                    if sounds[part]:
                                        sounds[part].play()

                # Draw translucent hitboxes
                for part, (x, y) in [("left", (left_shoulder_x, left_shoulder_y)),
                                     ("right", (right_shoulder_x, right_shoulder_y)),
                                     ("head", (head_x, head_y)),
                                     ("chest", (chest_x, chest_y)),
                                     ("hip", (hip_x, hip_y))]:
                    cv2.circle(overlay, (x, y), radius, colors[part], -1)
                    cv2.putText(frame, f"{part} sound", (x - radius, y - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # Display the frame for the current camera
            cv2.imshow(f"Player {idx + 1} - Camera", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release all cameras and close all windows
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
