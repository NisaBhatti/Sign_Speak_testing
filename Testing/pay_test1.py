# Save as: model_testing/test_pay_robust.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os

# ========== CONFIGURATION ==========
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_pay\pay_robust.h5"
CONFIDENCE_THRESHOLD = 0.85

print("="*60)
print("PAY (پ) DETECTION - VERTICAL ONLY")
print("="*60)

# ========== NORMALIZATION (MUST MATCH TRAINING) ==========
def normalize_landmarks_keep_orientation(features):
    pts = np.array(features).reshape(21, 2)
    wrist = pts[0]
    pts = pts - wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    # NO rotation
    return pts.flatten()

# ========== LOAD MODEL ==========
print("\nLoading Pay model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========== WEBCAM ==========
cap = cv2.VideoCapture(0)
print("\nStarting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    status = "No Hand Detected"
    color = (128, 128, 128)
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            features = []
            for lm in hand_landmarks.landmark:
                features.append(lm.x)
                features.append(lm.y)

            # Apply SAME normalization as training (no rotation)
            normalized = normalize_landmarks_keep_orientation(features)
            normalized = (normalized + 1.0) / 2.0
            normalized = np.clip(normalized, 0, 1)

            input_data = np.expand_dims(normalized, axis=0)
            prediction = model.predict(input_data, verbose=0)[0][0]
            confidence = float(prediction)

            if confidence >= CONFIDENCE_THRESHOLD:
                status = "PAY (پ) DETECTED"
                color = (0, 255, 0)
            else:
                status = f"Not Pay ({confidence*100:.1f}%)"
                color = (0, 0, 255)

    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if results.multi_hand_landmarks:
        bar_width = int(confidence * 200)
        cv2.rectangle(frame, (w - 220, 15), (w - 20, 45), (50, 50, 50), -1)
        cv2.rectangle(frame, (w - 220, 15), (w - 220 + bar_width, 45), color, -1)
        cv2.putText(frame, f"{confidence*100:.0f}%", (w - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Pay (پ) - Vertical Only Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nDetection stopped.")