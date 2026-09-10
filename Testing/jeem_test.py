# Save as: D:\MODEL\Sign_Speak_testing-main\Testing\test_jeem_robust.py

import os
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 60)
print("SIGNSPEAK - Jeem Robust Detection")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = r"D:\MODEL\Sign_Speak_testing-main"
MODEL_PATH = os.path.join(BASE_DIR, "Exported_Model", "jeem_robust.tflite")
INFO_PATH  = os.path.join(BASE_DIR, "Exported_Model", "jeem_robust_info.json")

# ============================================================
# LOAD MODEL + INFO
# ============================================================
print("\n📁 Loading model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun jeem_train.py first.")

if not os.path.isfile(INFO_PATH):
    raise FileNotFoundError(f"Info file not found: {INFO_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

THRESHOLD = float(info.get("threshold", 0.5))

print(f"   Model     : {info.get('model', 'jeem_robust')}")
print(f"   Sign      : {info.get('sign', 'ج')}  ({info.get('sign_name', 'Jeem')})")
print(f"   Accuracy  : {info.get('test_accuracy', 0) * 100:.1f}%")
print(f"   AUC       : {info.get('test_auc', 0):.4f}")
print(f"   Precision : {info.get('test_precision', 0) * 100:.1f}%")
print(f"   Recall    : {info.get('test_recall', 0) * 100:.1f}%")
print(f"   Threshold : {THRESHOLD}")

# ============================================================
# MEDIAPIPE
# ============================================================
print("\n🖐️ Starting MediaPipe...")
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# CAMERA
# ============================================================
print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN = "Jeem Detection - Robust"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1100, 800)

print("\n" + "=" * 60)
print("🎯 JEEM DETECTION")
print("=" * 60)
print("   GREEN dots = YOUR hand")
print("   GREEN text = Jeem detected!")
print("")
print("   How to sign Jeem (ج):")
print("   ☝️ Index finger curved like a hook (like a tight 'C')")
print("   ✊ Other fingers folded into palm")
print("")
print("   Q=Quit  S=Save Screenshot")
print("=" * 60 + "\n")

# ============================================================
# SMOOTHING
# ============================================================
prediction_history = deque(maxlen=10)
frame_count = 0

def get_landmark_features(hand_landmarks):
    """Return (1, 42) float32 array from MediaPipe landmarks."""
    feats = []
    for lm in hand_landmarks.landmark:
        feats.extend([lm.x, lm.y])
    return np.array(feats, dtype=np.float32).reshape(1, -1)

def predict_jeem(features):
    """Run TFLite inference, return probability of Jeem (0-1)."""
    interpreter.set_tensor(input_details[0]["index"], features)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]["index"])[0][0])

# ============================================================
# MAIN LOOP
# ============================================================
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

            # ---- Draw YOUR hand (green) ----
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 8, (0, 200, 0), 2)

            for c in mp_hands.HAND_CONNECTIONS:
                s = hand_landmarks.landmark[c[0]]
                e = hand_landmarks.landmark[c[1]]
                cv2.line(
                    frame,
                    (int(s.x * w), int(s.y * h)),
                    (int(e.x * w), int(e.y * h)),
                    (0, 255, 0), 2
                )

            # ---- Predict ----
            feats = get_landmark_features(hand_landmarks)
            prediction = predict_jeem(feats)

            prediction_history.append(prediction)
            smooth = sum(prediction_history) / len(prediction_history)

            is_jeem = smooth > THRESHOLD

            # ---- Header ----
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

            if is_jeem:
                if smooth > 0.85:
                    text, color = "✅ JEEM (ج) - HIGH CONFIDENCE!", (0, 255, 0)
                elif smooth > 0.70:
                    text, color = "✅ JEEM (ج) - Good", (0, 255, 128)
                else:
                    text, color = "✅ JEEM (ج) - Low confidence", (0, 255, 255)
            else:
                if smooth < 0.20:
                    text, color = "❌ NOT JEEM - Very different", (0, 0, 255)
                elif smooth < 0.40:
                    text, color = "❌ NOT JEEM - Different", (0, 128, 255)
                else:
                    text, color = "⚠️ UNCERTAIN - Close to Jeem", (0, 165, 255)

            cv2.putText(frame, text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Jeem Score: {smooth * 100:.1f}%", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(
                frame,
                f"Raw: {prediction:.4f} | Smooth: {smooth:.4f} | Thr: {THRESHOLD:.2f}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1
            )

            # ---- Legend ----
            cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
            cv2.circle(frame, (30, h - 20), 6, (0, 255, 0), -1)
            cv2.putText(frame, "Your Hand", (45, h - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Q=Quit  S=Save", (w - 200, h - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    else:
        # ---- No hand ----
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        prediction_history.clear()

    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        fn = f"jeem_test_{frame_count}.png"
        cv2.imwrite(fn, frame)
        print(f"📸 Saved: {fn}")

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Done! Processed {frame_count} frames")