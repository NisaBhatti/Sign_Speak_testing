# Save as: model_testing/test_tey_live.py
"""
Live webcam test for the Tey (ٹ) model.
Detects ONLY the Tey sign (fist + thumb up). Everything else → "Not Tey".
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# ========== CONFIG ==========
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_tey\tey_robust.tflite"
THRESHOLD = 0.5   # confidence threshold for detecting Tey

# ========== LOAD TFLITE ==========
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"   Input shape:  {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

print("\n✅ Running. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label_text = "No hand detected"
    color = (128, 128, 128)
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Build 42-feature vector (already normalized 0-1 by MediaPipe)
        features = []
        for lm in hand.landmark:
            features.append(lm.x)
            features.append(lm.y)
        features = np.array([features], dtype=np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

        confidence = float(pred)
        if pred >= THRESHOLD:
            label_text = f"TEY (ٹ)  {pred*100:.1f}%"
            color = (0, 255, 0)
        else:
            label_text = f"Not Tey  {(1-pred)*100:.1f}%"
            color = (0, 0, 255)

    # Overlay
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, f"Threshold: {THRESHOLD}", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("SignSpeak - Tey (ٹ) Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")