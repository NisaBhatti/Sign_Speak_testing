# Save as: testing/test_kaf_robust.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import sqlite3
import re
from collections import deque

print("="*60)
print("SIGNSPEAK - Kāf (ک) Robust Detection")
print("="*60)

# ========== PATHS ==========
MODEL_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_kaf\kaf_robust.tflite"
INFO_PATH  = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_kaf\kaf_robust_info.json"
DB_PATH    = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\main_dataset.db"

# Urdu/Persian letter Kāf (Urdu K) — U+06A9
KAF_LETTER = 'ک'

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

# ========== GET REFERENCE KAF FROM DATABASE ==========
print("\n📊 Loading reference Kāf landmarks...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM rightHandDataset")
all_data = cursor.fetchall()
conn.close()

kaf_samples = []
for row in all_data:
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    if label == KAF_LETTER:
        features = [float(row[i]) for i in range(1, 43)]
        kaf_samples.append(features)

if len(kaf_samples) == 0:
    print(f"\n❌ ERROR: No reference samples found for '{KAF_LETTER}' (U+06A9)!")
    print(f"   The DB may not contain Kāf samples.")
    exit(1)

kaf_samples = np.array(kaf_samples, dtype=np.float32)
kaf_samples[:, 0::2] = kaf_samples[:, 0::2] / 300.0
kaf_samples[:, 1::2] = kaf_samples[:, 1::2] / 300.0

ref_kaf = kaf_samples.mean(axis=0).reshape(21, 2)
print(f"   Reference from {len(kaf_samples)} Kāf samples")

# ============================================================
# ⭐ Normalize reference to 0-1 bounding box
# ============================================================
ref_min = ref_kaf.min(axis=0)
ref_max = ref_kaf.max(axis=0)
ref_range = np.maximum(ref_max - ref_min, 1e-6)
ref_kaf_norm = (ref_kaf - ref_min) / ref_range

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

cv2.namedWindow('Kaf Detection - Robust', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Kaf Detection - Robust', 1100, 800)

print("\n" + "="*60)
print("🎯 KĀF (ک) DETECTION")
print("="*60)
print("   GREEN dots = YOUR hand")
print("   WHITE skeleton = Reference Kāf (follows your hand)")
print("   GREEN text = Kāf detected!")
print("")
print("   How to sign Kāf (ک):")
print("   Follow the handshape shown in the reference skeleton")
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

    # Guide lines for ideal hand zone
    cv2.line(frame, (0, int(h*0.15)), (w, int(h*0.15)), (60, 60, 60), 1)
    cv2.line(frame, (0, int(h*0.85)), (w, int(h*0.85)), (60, 60, 60), 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # ===== HAND BOUNDING BOX =====
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            hand_center_x = (min(xs) + max(xs)) / 2
            hand_center_y = (min(ys) + max(ys)) / 2
            hand_size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

            # ⭐ Anchor reference to hand
            ref_world = (ref_kaf_norm - 0.5) * hand_size
            ref_world[:, 0] += hand_center_x
            ref_world[:, 1] += hand_center_y

            # ⭐⭐ CLAMP inside frame with margin
            margin = 0.05
            rmin = ref_world.min(axis=0)
            rmax = ref_world.max(axis=0)
            if rmin[0] < margin:
                ref_world[:, 0] += (margin - rmin[0])
            if rmax[0] > 1.0 - margin:
                ref_world[:, 0] -= (rmax[0] - (1.0 - margin))
            if rmin[1] < margin:
                ref_world[:, 1] += (margin - rmin[1])
            if rmax[1] > 1.0 - margin:
                ref_world[:, 1] -= (rmax[1] - (1.0 - margin))

            ref_px = np.stack([
                (ref_world[:, 0] * w).astype(int),
                (ref_world[:, 1] * h).astype(int)
            ], axis=1)

            # ===== DRAW REFERENCE (WHITE) =====
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

            is_kaf = smooth_pred > 0.5

            # ===== DISPLAY =====
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

            if is_kaf:
                if smooth_pred > 0.8:
                    text = f"✅ KĀF (ک) - HIGH CONFIDENCE!"
                    color = (0, 255, 0)
                elif smooth_pred > 0.6:
                    text = f"✅ KĀF (ک) - Good"
                    color = (0, 255, 128)
                else:
                    text = f"✅ KĀF (ک) - Low confidence"
                    color = (0, 255, 255)
            else:
                if smooth_pred < 0.2:
                    text = f"❌ NOT KĀF - Very different"
                    color = (0, 0, 255)
                elif smooth_pred < 0.4:
                    text = f"❌ NOT KĀF - Different"
                    color = (0, 128, 255)
                else:
                    text = f"⚠️ UNCERTAIN - Close to Kāf"
                    color = (0, 165, 255)

            cv2.putText(frame, text, (20, 35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Kāf Score: {smooth_pred*100:.1f}%", (20, 65),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Raw: {prediction:.4f} | Smoothed: {smooth_pred:.4f}", (20, 85),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Bottom legend
            cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.circle(frame, (30, h-20), 6, (255, 255, 255), -1)
            cv2.putText(frame, "Reference Kāf", (45, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (200, h-20), 6, (0, 255, 0), -1)
            cv2.putText(frame, "Your Hand", (215, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Q=Quit S=Save", (w-180, h-13),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    else:
        # No hand — ghost reference in center
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        ghost_size = 0.5
        ghost_center = np.array([0.5, 0.5])
        ref_ghost = (ref_kaf_norm - 0.5) * ghost_size + ghost_center
        for conn in mp_hands.HAND_CONNECTIONS:
            s = ref_ghost[conn[0]]
            e = ref_ghost[conn[1]]
            cv2.line(frame,
                    (int(s[0]*w), int(s[1]*h)),
                    (int(e[0]*w), int(e[1]*h)),
                    (80, 80, 80), 1)
        for (rx, ry) in ref_ghost:
            cv2.circle(frame, (int(rx*w), int(ry*h)), 4, (120, 120, 120), -1)

    cv2.imshow('Kaf Detection - Robust', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"kaf_test_{frame_count}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Done! Processed {frame_count} frames")