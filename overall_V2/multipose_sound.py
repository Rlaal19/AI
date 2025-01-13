import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pygame
import numpy as np

# โหลด MoveNet Module
model_path = '/Users/parichaya23icloud.com/Desktop/AI/overall_V2/movenet-tensorflow2-multipose-lightning-v1'
model = hub.load(model_path)
movenet = model.signatures['serving_default']

# เริ่มต้น pygame mixer และกำหนดจำนวน Channel
pygame.mixer.init()
pygame.mixer.set_num_channels(5)  # จำกัดจำนวน Channels สูงสุดให้ 5

# ฟังก์ชันเล่นเสียงใน Channel ที่กำหนด
def play_sound_limited_with_channel(file, channel_id):
    """เล่นเสียงใน Channel ที่กำหนดตาม ID ของคน"""
    max_channels = 5  # กำหนดจำนวน Channel สูงสุด
    channel_id %= max_channels  # จำกัด ID ไม่ให้เกินจำนวน Channel
    channel = pygame.mixer.Channel(channel_id)
    if not channel.get_busy():  # ถ้า Channel ว่าง
        sound = pygame.mixer.Sound(file)
        channel.play(sound)  # เล่นเสียงใน Channel นี้
    else:
        # ถ้า Channel ไม่ว่าง ให้หยุด Channel ปัจจุบันและเล่นเสียงใหม่
        channel.stop()
        sound = pygame.mixer.Sound(file)
        channel.play(sound)

# ฟังก์ชันตรวจจับท่าทางที่ต้องการ
def is_pose_detected(keypoints, confidence_threshold):
    """กำหนดเงื่อนไขสำหรับท่าทาง"""
    left_wrist = keypoints[9]
    left_elbow = keypoints[7]
    right_wrist = keypoints[10]
    right_elbow = keypoints[8]
    hip = keypoints[11]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    lw_y, lw_x, lw_conf = left_wrist
    le_y, le_x, le_conf = left_elbow
    rw_y, rw_x, rw_conf = right_wrist
    re_y, re_x, re_conf = right_elbow
    hip_y, hip_x, hip_conf = hip
    la_y, la_x, la_conf = left_ankle
    ra_y, ra_x, ra_conf = right_ankle
    lk_y, lk_x, lk_conf = left_knee
    rk_y, rk_x, rk_conf = right_knee

    # ตรวจจับท่าทาง: ข้อมือซ้ายอยู่เหนือศอกซ้าย
    if lw_conf > confidence_threshold and le_conf > confidence_threshold:
        if lw_y < le_y:
            return "left_hand_up"

    # ตรวจจับท่าทาง: ข้อมือขวาอยู่เหนือศอกขวา
    if rw_conf > confidence_threshold and re_conf > confidence_threshold:
        if rw_y < re_y:
            return "right_hand_up"

    # ตรวจจับท่าทาง: ยืนตรง
    if hip_conf > confidence_threshold:
        if abs(hip_y - re_y) < 50 and abs(hip_y - le_y) < 50:
            return "standing"

    # ตรวจจับท่าทาง: การนั่ง
    if hip_conf > confidence_threshold:
        if hip_y > la_y and hip_y > ra_y:
            return "sit"

    # ตรวจจับท่าทาง: มือทั้งสองข้างแตะเข่าทั้งสองข้าง
    if lw_conf > confidence_threshold and rw_conf > confidence_threshold and lk_conf > confidence_threshold and rk_conf > confidence_threshold:
        if abs(lw_y - lk_y) < 100 and abs(rw_y - rk_y) < 100:
            return "both_hands_knee"

    return None

# ฟังก์ชันวาด Keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)

# Dictionary สำหรับกำหนดเส้นเชื่อมต่อ
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

# ฟังก์ชันวาด Connections
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

# ฟังก์ชันวนลูปตรวจจับทุกคนในเฟรม
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    person_id = 0  # ใช้ตัวระบุ ID ของคนในเฟรม
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

        # ตรวจจับท่าทางและเล่นเสียงเมื่อพบ
        pose = is_pose_detected(person, confidence_threshold)
        if pose == "left_hand_up":
            play_sound_limited_with_channel('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_Dead Kick_Press_0115.6.wav', person_id)
        elif pose == "right_hand_up":
            play_sound_limited_with_channel('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/DI_HiHat_Foot_1144.4.wav', person_id)
        elif pose == "standing":
            play_sound_limited_with_channel('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Overhead Sample 4.wav', person_id)
        elif pose == "sit":
            play_sound_limited_with_channel('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Snare Sample 27.wav', person_id)
        elif pose == "both_hands_knee":
            play_sound_limited_with_channel('/Users/parichaya23icloud.com/Desktop/AI/overall_V2/used_sound_file/Tom Sample 17.wav', person_id)

        person_id += 1  # เพิ่ม ID สำหรับคนถัดไป

# เริ่มต้น Webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image สำหรับโมเดล
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
    input_img = tf.cast(img, dtype=tf.int32)

    # ตรวจจับ keypoints
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

    # วาดผลลัพธ์ลงบน frame
    loop_through_people(frame, keypoints_with_scores, EDGES, confidence_threshold=0.3)

    cv2.imshow('MoveNet Multipose', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
