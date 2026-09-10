# Save as: testing/test_toayn_robust.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import sqlite3
import re
from collections import deque

print("="*60)
print("SIGNSPEAK - Toayn Robust Detection")
print("="*60)

# ========== PATHS ==========
MODEL_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_toayn\toayn_robust.tflite"
INFO_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_toayn\toayn_robust_info.json"
DB_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\main_dataset.db"

# Arabic letter for Toayn
TOAYN_LETTER = 'ظ'

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

# ========== GET REFERENCE TOAYN FROM DATABASE ==========
print("\n📊 Loading reference Toayn landmarks...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM rightHandDataset")
all_data = cursor.fetchall()
conn.close()

toayn_samples = []
for row in all_data:
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    if label == TOAYN_LETTER:
        features = [float(row[i]) for i in range(1, 43)]
        toayn_samples.append(features)

toayn_samples = np.array(toayn_samples, dtype=np.float32)
# Normalize to 0-1
toayn_samples[:, 0::2] = toayn_samples[:, 0::2] / 300.0
toayn_samples[:, 1::2] = toayn_samples[:, 1::2] / 300.0

# Average Toayn landmarks
ref_toayn = toayn_samples.mean(axis=0).reshape(21, 2)
print(f"   Reference from {len(toayn_samples)} Toayn samples")

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

cv2.namedWindow('Toayn Detection - Robust', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Toayn Detection - Robust', 1100, 800)

print("\n" + "="*60)
print("🎯 TOAYN DETECTION")
print("="*60)
print("   GREEN dots = YOUR hand")
print("   WHITE skeleton = Reference Toayn")
print("   GREEN text = Toayn detected!")
print("")
print("   How to sign Toayn (ظ):")
print("   ☝️ Index finger straight UP")
print("   👍 Thumb folded across palm")
print("   ✊ Other fingers folded")
print("   🖐️ Any hand position works!")
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
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # ===== DRAW REFERENCE TOAYN (WHITE) =====
            for i, (rx, ry) in enumerate(ref_toayn):
                x = int(rx * w)
                y = int(ry * h)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
            
            # Reference connections
            for conn in mp_hands.HAND_CONNECTIONS:
                s_x = int(ref_toayn[conn[0]][0] * w)
                s_y = int(ref_toayn[conn[0]][1] * h)
                e_x = int(ref_toayn[conn[1]][0] * w)
                e_y = int(ref_toayn[conn[1]][1] * h)
                cv2.line(frame, (s_x, s_y), (e_x, e_y), (255, 255, 255), 1)
            
            # ===== DRAW YOUR HAND (GREEN) =====
            for lm in hand_landmarks.landmark:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 8, (0, 200, 0), 2)
            
            # Your connections
            for conn in mp_hands.HAND_CONNECTIONS:
                s = hand_landmarks.landmark[conn[0]]
                e = hand_landmarks.landmark[conn[1]]
                cv2.line(frame, 
                       (int(s.x*w), int(s.y*h)),
                       (int(e.x*w), int(e.y*h)),
                       (0, 255, 0), 2)
            
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
            
            is_toayn = smooth_pred > 0.5
            confidence = smooth_pred if is_toayn else (1 - smooth_pred)
            
            # ===== DISPLAY =====
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            
            if is_toayn:
                if smooth_pred > 0.8:
                    text = f"✅ TOAYN (ظ) - HIGH CONFIDENCE!"
                    color = (0, 255, 0)
                elif smooth_pred > 0.6:
                    text = f"✅ TOAYN (ظ) - Good"
                    color = (0, 255, 128)
                else:
                    text = f"✅ TOAYN (ظ) - Low confidence"
                    color = (0, 255, 255)
            else:
                if smooth_pred < 0.2:
                    text = f"❌ NOT TOAYN - Very different"
                    color = (0, 0, 255)
                elif smooth_pred < 0.4:
                    text = f"❌ NOT TOAYN - Different"
                    color = (0, 128, 255)
                else:
                    text = f"⚠️ UNCERTAIN - Close to Toayn"
                    color = (0, 165, 255)
            
            cv2.putText(frame, text, (20, 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Toayn Score: {smooth_pred*100:.1f}%", (20, 65),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Raw: {prediction:.4f} | Smoothed: {smooth_pred:.4f}", (20, 85),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Bottom legend
            cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.circle(frame, (30, h-20), 6, (255, 255, 255), -1)
            cv2.putText(frame, "Reference Toayn", (45, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (200, h-20), 6, (0, 255, 0), -1)
            cv2.putText(frame, "Your Hand", (215, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Q=Quit S=Save", (w-180, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    else:
        # No hand detected
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        # Show reference in gray
        for i, (rx, ry) in enumerate(ref_toayn):
            x, y = int(rx * w), int(ry * h)
            cv2.circle(frame, (x, y), 4, (100, 100, 100), -1)
    
    cv2.imshow('Toayn Detection - Robust', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"toayn_test_{frame_count}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Done! Processed {frame_count} frames")