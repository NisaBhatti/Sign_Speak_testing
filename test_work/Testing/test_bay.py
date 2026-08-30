# Save as: testing/test_bay_robust.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import sqlite3
import re
from collections import deque

print("="*60)
print("SIGNSPEAK - Bay Robust Detection")
print("="*60)

# ========== PATHS ==========
MODEL_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_bay\bay_robust.tflite"
INFO_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_bay\bay_robust_info.json"
DB_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\main_dataset.db"

# ========== LOAD MODEL ==========
print("\n📁 Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(INFO_PATH, 'r', encoding='utf-8') as f:
    info = json.load(f)

print(f"   Model: {info['model']}")
print(f"   Accuracy: {info['test_accuracy']*100:.1f}%")
print(f"   AUC: {info['test_auc']:.4f}")
print(f"   Threshold: {info['threshold']}")

# ========== GET REFERENCE BAY FROM DATABASE ==========
print("\n📊 Loading reference Bay landmarks...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM rightHandDataset")
all_data = cursor.fetchall()
conn.close()

bay_samples = []
for row in all_data:
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    if label == 'ب':
        features = [float(row[i]) for i in range(1, 43)]
        bay_samples.append(features)

bay_samples = np.array(bay_samples, dtype=np.float32)

if len(bay_samples) == 0:
    print("❌ ERROR: No Bay samples found in database!")
    print("   Please add Bay samples first.")
    exit(1)

# Normalize to 0-1
bay_samples[:, 0::2] = bay_samples[:, 0::2] / 300.0
bay_samples[:, 1::2] = bay_samples[:, 1::2] / 300.0

# Average Bay landmarks
ref_bay = bay_samples.mean(axis=0).reshape(21, 2)
print(f"   Reference from {len(bay_samples)} Bay samples")

# ========== NORMALIZE REFERENCE SKELETON ==========
# Scale down and center the reference so it fits nicely on screen
ref_center_x = ref_bay[:, 0].mean()
ref_center_y = ref_bay[:, 1].mean()

# Center the landmarks
ref_centered = ref_bay.copy()
ref_centered[:, 0] -= ref_center_x
ref_centered[:, 1] -= ref_center_y

# Scale to a fixed size (0.15 of screen width/height)
ref_scale = 0.15
ref_max_range = max(ref_centered[:, 0].max() - ref_centered[:, 0].min(),
                    ref_centered[:, 1].max() - ref_centered[:, 1].min())
if ref_max_range > 0:
    ref_centered *= (ref_scale / ref_max_range)

# Position at a fixed location on screen (center-right area)
ref_offset_x = 0.65  # 65% from left
ref_offset_y = 0.40  # 40% from top

ref_bay_normalized = ref_centered.copy()
ref_bay_normalized[:, 0] += ref_offset_x
ref_bay_normalized[:, 1] += ref_offset_y

print(f"   Reference normalized to fixed size and position")

# ========== MEDIAPIPE ==========
print("\n🖐️ Starting MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========== CAMERA ==========
print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Bay Detection - Robust', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Bay Detection - Robust', 1100, 800)

print("\n" + "="*60)
print("🎯 BAY DETECTION")
print("="*60)
print("   GREEN dots = YOUR hand (matches Bay)")
print("   YELLOW dots = YOUR hand (different sign)")
print("   WHITE skeleton = Reference Bay")
print("   GREEN text = Bay detected!")
print("")
print("   How to sign Bay (ب):")
print("   ✋ Four fingers straight and together")
print("   👍 Thumb curved (tucked in)")
print("   🖐️ Palm facing forward")
print("")
print("   Q=Quit  S=Save Screenshot")
print("="*60 + "\n")

# ========== SMOOTHING ==========
prediction_history = deque(maxlen=10)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # ===== DRAW REFERENCE BAY (ALWAYS VISIBLE, WHITE, SMALL) =====
    ref_color = (255, 255, 255)
    ref_alpha = 0.6  # slightly transparent
    
    # Create overlay for reference
    overlay = frame.copy()
    
    for i, (rx, ry) in enumerate(ref_bay_normalized):
        x = int(rx * w)
        y = int(ry * h)
        cv2.circle(overlay, (x, y), 4, ref_color, -1)
        cv2.circle(overlay, (x, y), 6, ref_color, 1)
    
    # Reference connections (only main skeleton, not all connections)
    # Simplified: connect wrist to each finger base, and finger joints
    ref_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # index
        (5, 9), (9, 10), (10, 11), (11, 12),  # middle (from index base)
        (9, 13), (13, 14), (14, 15), (15, 16), # ring
        (13, 17), (17, 18), (18, 19), (19, 20), # pinky
        (0, 17),                               # wrist to pinky base
    ]
    
    for conn in ref_connections:
        s_x = int(ref_bay_normalized[conn[0]][0] * w)
        s_y = int(ref_bay_normalized[conn[0]][1] * h)
        e_x = int(ref_bay_normalized[conn[1]][0] * w)
        e_y = int(ref_bay_normalized[conn[1]][1] * h)
        cv2.line(overlay, (s_x, s_y), (e_x, e_y), ref_color, 1)
    
    # Blend reference overlay
    cv2.addWeighted(overlay, ref_alpha, frame, 1 - ref_alpha, 0, frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # ===== PREDICT =====
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y])
            
            features = np.array(features, dtype=np.float32).reshape(1, -1)
            
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # Smooth
            prediction_history.append(prediction)
            smooth_pred = sum(prediction_history) / len(prediction_history)
            
            is_bay = smooth_pred > 0.5
            confidence = smooth_pred if is_bay else (1 - smooth_pred)
            
            # ===== DRAW YOUR HAND =====
            hand_color = (0, 255, 0) if is_bay else (0, 255, 255)
            hand_outline = (0, 200, 0) if is_bay else (0, 200, 200)
            
            for lm in hand_landmarks.landmark:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 6, hand_color, -1)
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 8, hand_outline, 2)
            
            # Your connections
            for conn in mp_hands.HAND_CONNECTIONS:
                s = hand_landmarks.landmark[conn[0]]
                e = hand_landmarks.landmark[conn[1]]
                cv2.line(frame, 
                       (int(s.x*w), int(s.y*h)),
                       (int(e.x*w), int(e.y*h)),
                       hand_color, 2)
            
            # ===== DISPLAY =====
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            
            if is_bay:
                if smooth_pred > 0.8:
                    text = f"✅ BAY (ب) - HIGH CONFIDENCE!"
                    color = (0, 255, 0)
                elif smooth_pred > 0.6:
                    text = f"✅ BAY (ب) - Good"
                    color = (0, 255, 128)
                else:
                    text = f"✅ BAY (ب) - Low confidence"
                    color = (0, 255, 255)
            else:
                if smooth_pred < 0.2:
                    text = f"❌ NOT BAY - Very different"
                    color = (0, 0, 255)
                elif smooth_pred < 0.4:
                    text = f"❌ NOT BAY - Different"
                    color = (0, 128, 255)
                else:
                    text = f"⚠️ UNCERTAIN - Close to Bay"
                    color = (0, 165, 255)
            
            cv2.putText(frame, text, (20, 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Bay Score: {smooth_pred*100:.1f}%", (20, 65),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Raw: {prediction:.4f} | Smoothed: {smooth_pred:.4f}", (20, 85),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Bottom legend
            cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.circle(frame, (30, h-20), 6, (255, 255, 255), -1)
            cv2.putText(frame, "Reference Bay", (45, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if is_bay:
                cv2.circle(frame, (200, h-20), 6, (0, 255, 0), -1)
                cv2.putText(frame, "Your Hand (MATCH!)", (215, h-13),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (200, h-20), 6, (0, 255, 255), -1)
                cv2.putText(frame, "Your Hand (Different)", (215, h-13),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(frame, f"Q=Quit S=Save", (w-180, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    else:
        # No hand detected
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    
    cv2.imshow('Bay Detection - Robust', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"bay_test_{frame_count}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Done! Processed {frame_count} frames")