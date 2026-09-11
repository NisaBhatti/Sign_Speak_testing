# Save as: testing/test_rray_robust.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import sqlite3
import re
from collections import deque

print("="*60)
print("SIGNSPEAK - Rray (Ṛay) Robust Detection")
print("="*60)

# ========== PATHS ==========
MODEL_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_rray\rray_robust.tflite"
INFO_PATH  = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_rray\rray_robust_info.json"
DB_PATH    = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\main_dataset.db"

# Urdu/Punjabi Retroflex R
RRAY_LETTER = 'ڑ'   # U+0691

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

# ========== GET REFERENCE RRAY FROM DATABASE ==========
print("\n📊 Loading reference Rray landmarks...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM rightHandDataset")
all_data = cursor.fetchall()
conn.close()

rray_samples = []
for row in all_data:
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    if label == RRAY_LETTER:
        features = [float(row[i]) for i in range(1, 43)]
        rray_samples.append(features)

rray_samples = np.array(rray_samples, dtype=np.float32)
rray_samples[:, 0::2] = rray_samples[:, 0::2] / 300.0
rray_samples[:, 1::2] = rray_samples[:, 1::2] / 300.0

ref_rray = rray_samples.mean(axis=0).reshape(21, 2)
print(f"   Reference from {len(rray_samples)} Rray samples")

# ============================================================
# ⭐ Normalize reference to 0-1 bounding box
# ============================================================
ref_min = ref_rray.min(axis=0)
ref_max = ref_rray.max(axis=0)
ref_range = np.maximum(ref_max - ref_min, 1e-6)
ref_rray_norm = (ref_rray - ref_min) / ref_range

print(f"   Reference normalized to bounding box")

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

cv2.namedWindow('Rray Detection - Robust', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Rray Detection - Robust', 1100, 800)

print("\n" + "="*60)
print("🎯 RRAY (ṚAY) DETECTION")
print("="*60)
print("   GREEN dots = YOUR hand")
print("   WHITE skeleton = Reference Rray (follows your hand)")
print("   GREEN text = Rray detected!")
print("")
print("   How to sign Rray (ڑ):")
print("   ☝️ Index finger straight UP")
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
            
            # ===== COMPUTE YOUR HAND BOUNDING BOX =====
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            hand_center_x = (min(xs) + max(xs)) / 2
            hand_center_y = (min(ys) + max(ys)) / 2
            hand_size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
            
            # ⭐ Anchor reference to YOUR hand position and scale
            ref_world = (ref_rray_norm - 0.5) * hand_size
            ref_world[:, 0] += hand_center_x
            ref_world[:, 1] += hand_center_y
            
            ref_px = np.stack([
                (ref_world[:, 0] * w).astype(int),
                (ref_world[:, 1] * h).astype(int)
            ], axis=1)
            
            # ===== DRAW REFERENCE RRAY (WHITE) =====
            for conn in mp_hands.HAND_CONNECTIONS:
                s = ref_px[conn[0]]
                e = ref_px[conn[1]]
                cv2.line(frame, tuple(s), tuple(e), (255, 255, 255), 1)
            for (x, y) in ref_px:
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
            
            # ===== DRAW YOUR HAND (GREEN) =====
            for lm in hand_landmarks.landmark:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 8, (0, 200, 0), 2)
            
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
            
            prediction_history.append(prediction)
            smooth_pred = sum(prediction_history) / len(prediction_history)
            
            is_rray = smooth_pred > 0.5
            confidence = smooth_pred if is_rray else (1 - smooth_pred)
            
            # ========== DISPLAY ==========
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
            
            if is_rray:
                if smooth_pred > 0.8:
                    text = f"✅ RRAY (ڑ) - HIGH CONFIDENCE!"
                    color = (0, 255, 0)
                elif smooth_pred > 0.6:
                    text = f"✅ RRAY (ڑ) - Good"
                    color = (0, 255, 128)
                else:
                    text = f"✅ RRAY (ڑ) - Low confidence"
                    color = (0, 255, 255)
            else:
                if smooth_pred < 0.2:
                    text = f"❌ NOT RRAY - Very different"
                    color = (0, 0, 255)
                elif smooth_pred < 0.4:
                    text = f"❌ NOT RRAY - Different"
                    color = (0, 128, 255)
                else:
                    text = f"⚠️ UNCERTAIN - Close to Rray"
                    color = (0, 165, 255)
            
            cv2.putText(frame, text, (20, 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Rray Score: {smooth_pred*100:.1f}%", (20, 65),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Raw: {prediction:.4f} | Smoothed: {smooth_pred:.4f}", (20, 85),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Bottom legend
            cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.circle(frame, (30, h-20), 6, (255, 255, 255), -1)
            cv2.putText(frame, "Reference Rray", (45, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (200, h-20), 6, (0, 255, 0), -1)
            cv2.putText(frame, "Your Hand", (215, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Q=Quit S=Save", (w-180, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    else:
        # No hand — show ghost reference in center
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        ghost_size = 0.5
        ghost_center = np.array([0.5, 0.5])
        ref_ghost = (ref_rray_norm - 0.5) * ghost_size + ghost_center
        for conn in mp_hands.HAND_CONNECTIONS:
            s = ref_ghost[conn[0]]
            e = ref_ghost[conn[1]]
            cv2.line(frame,
                    (int(s[0]*w), int(s[1]*h)),
                    (int(e[0]*w), int(e[1]*h)),
                    (80, 80, 80), 1)
        for (rx, ry) in ref_ghost:
            cv2.circle(frame, (int(rx*w), int(ry*h)), 4, (120, 120, 120), -1)
    
    cv2.imshow('Rray Detection - Robust', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"rray_test_{frame_count}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Done! Processed {frame_count} frames")