import cv2
import numpy as np
import mediapipe as mp
import math
import pygame
import time
import random

# Initialize Mediapipe Holistic and Hands
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Pygame display for particle effects
screen_width = 1200
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Effects")

particles = []
# Load sound effects
sounds = {
    "left": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_Dead Kick_Press_0115.6.wav'),
    "right": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_HiHat_Foot_1144.4.wav'),
    "head": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Overhead Sample 4.wav'),
    "chest": pygame.mixer.Sound('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Snare Sample 27.wav'),
}

cap = cv2.VideoCapture(0)

properties = {
    "Frame Width": cv2.CAP_PROP_FRAME_WIDTH,
    "Frame Height": cv2.CAP_PROP_FRAME_HEIGHT,
    "FPS": cv2.CAP_PROP_FPS,
}

for prop_name, prop_id in properties.items():
    value = cap.get(prop_id)
    print(f"Bef setting: {prop_name}: {value}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

for prop_name, prop_id in properties.items():
    value = cap.get(prop_id)
    print(f"Aft setting: {prop_name}: {value}")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Activation flag
program_active = False

# Particle class
class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(3, 8)
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.life = random.randint(10, 25)
        self.speed = random.uniform(2, 6)
        self.angle = random.uniform(0, 2 * math.pi)
        self.gravity = 0.1
        self.dx = math.cos(self.angle) * self.speed
        self.dy = math.sin(self.angle) * self.speed

    def move(self):
        self.dy += self.gravity
        self.x += self.dx
        self.y += self.dy
        self.life -= 1

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)

    def is_alive(self):
        return self.life > 0

# Helper functions
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate the dynamic hitbox radius based on distance between shoulders
def calculate_dynamic_radius(shoulder_distance):
    base_distance = 0.4  # Base distance between shoulders in normalized coordinates
    base_radius = 100    # Base radius in pixels
    scaling_factor = shoulder_distance / base_distance
    return int(base_radius * scaling_factor)

def is_activation_pose_detected(results):
    if not results.pose_landmarks:
        return False
    
    # Get landmarks for wrists and shoulders
    left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

    # Check if both wrists are above shoulders (pose condition)
    return (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)

# Hand landmark indices for fingertips
fingertip_indices = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
]

# Main loop
current_hitbox = None
# last_hitbox = None
finger_inside_hitbox = False
last_toggle_time = 0 # time status
toggle_cooldown = 1  # delay time

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while True:
        # Capture each frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Holistic
        results = holistic.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        overlay = frame.copy()
        offset = 50

        # Check for activation pose
        if is_activation_pose_detected(results) and time.time() - last_toggle_time > toggle_cooldown:
            program_active = not program_active  # Toggle activation state
            last_toggle_time = time.time()  # record toggling time
            if program_active:
                print("Program activated!")
            else:
                print("Program deactivated!")
                
        if program_active:
            # Draw landmarks and process hitboxes
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Get the left and right shoulder positions
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                left_shoulder_x = int(left_shoulder.x * frame.shape[1])
                left_shoulder_y = int((left_shoulder.y - 0.1) * frame.shape[0])
                right_shoulder_x = int(right_shoulder.x * frame.shape[1])
                right_shoulder_y = int((right_shoulder.y - 0.1) * frame.shape[0])

                # Calculate dynamic radius for hitboxes
                shoulder_distance = calculate_distance(
                    (left_shoulder.x, left_shoulder.y),
                    (right_shoulder.x, right_shoulder.y)
                )
                radius = calculate_dynamic_radius(shoulder_distance)
                alpha = 0.4  # Transparency factor

                # Define positions for other body parts
                head = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                head_x = int(head.x * frame.shape[1])
                head_y = int((head.y - 0.2) * frame.shape[0])
                chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                chest_y = int((left_shoulder.y + right_shoulder.y + 0.3) / 2 * frame.shape[0])

                # Default colors for hitboxes
                left_color = (0, 255, 0)
                right_color = (0, 255, 0)
                head_color = (0, 255, 0)
                chest_color = (0, 255, 0)

                new_hitbox = None
                finger_detected = False
            
                screen.fill((0, 0, 0))  # Clear screen
                for particle in particles[:]:
                    particle.move()
                    particle.draw(screen)
                    if not particle.is_alive():
                        particles.remove(particle)
                pygame.display.flip()
                
                # Check for finger touching hitboxes
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
                                for _ in range(5):
                                    particles.append(Particle(left_shoulder_x, left_shoulder_y))

                            # Check if any finger touches the right shoulder circle
                            right_distance = calculate_distance((fingertip_x, fingertip_y), (right_shoulder_x, right_shoulder_y))
                            if right_distance < radius:
                                right_color = (0, 0, 255)  # Change to red when touched
                                new_hitbox = "right"
                                finger_detected = True
                                for _ in range(5):
                                    particles.append(Particle(right_shoulder_x, right_shoulder_y))

                            # Check if any finger touches the head circle
                            head_distance = calculate_distance((fingertip_x, fingertip_y), (head_x, head_y))
                            if head_distance < radius:
                                head_color = (0, 0, 255)  # Change to red when touched
                                new_hitbox = "head"
                                finger_detected = True
                                for _ in range(5):
                                    particles.append(Particle(head_x, head_y))

                            # Check if any finger touches the chest circle
                            chest_distance = calculate_distance((fingertip_x, fingertip_y), (chest_x, chest_y))
                            if chest_distance < radius:
                                chest_color = (0, 0, 255)  # Change to red when touched
                                new_hitbox = "chest"
                                finger_detected = True
                                for _ in range(5):
                                    particles.append(Particle(chest_x, chest_y))

                # Play sound if the finger is detected and conditions are met
                if finger_detected:
                    if new_hitbox != current_hitbox or not finger_inside_hitbox:
                        if new_hitbox in sounds:
                            sounds[new_hitbox].play()
                    finger_inside_hitbox = True
                else:
                    finger_inside_hitbox = False

                current_hitbox = new_hitbox

                # Draw translucent circles on shoulders, head, and chest
                cv2.circle(overlay, (left_shoulder_x, left_shoulder_y), radius, left_color, -1)
                cv2.circle(overlay, (right_shoulder_x, right_shoulder_y), radius, right_color, -1)
                cv2.circle(overlay, (head_x, head_y), radius, head_color, -1)
                cv2.circle(overlay, (chest_x, chest_y), radius, chest_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Display activation status
        activation_status = "Active" if program_active else "Inactive"
        status_color = (0, 255, 0) if program_active else (0, 0, 255)
        cv2.putText(frame, f"Program: {activation_status}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Pose-based Sound Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
pygame.quit()
cv2.destroyAllWindows()