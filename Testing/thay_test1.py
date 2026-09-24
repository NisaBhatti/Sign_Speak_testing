# Save as: D:\MODEL\Sign_Speak_testing-main\model_training\test_thay.py

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("="*60)
print("SIGNSPEAK - Thay (ث) REAL-TIME DETECTOR")
print("="*60)

# ========== CONFIG ==========
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\thay_model\thay_detector.h5"
THRESHOLD = 0.5
THAY_LETTER = 'ث'

# ========== LOAD MODEL ==========
print(f"\n📦 Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("   ✅ Model loaded")

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
print("\n🎥 Webcam started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    label_text = "No Hand"
    color = (128, 128, 128)
    confidence = 0.0
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Extract 42 features (x, y)
        features = []
        for lm in hand.landmark:
            features.append(lm.x)
            features.append(lm.y)
        
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Predict
        pred = model.predict(features, verbose=0)[0][0]
        confidence = float(pred)
        
        if pred > THRESHOLD:
            label_text = f"THAY ({THAY_LETTER}) - {pred*100:.1f}%"
            color = (0, 255, 0)  # Green
        else:
            label_text = f"Not Thay - {(1-pred)*100:.1f}%"
            color = (0, 0, 255)  # Red
    
    # Draw label banner
    cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
    cv2.putText(frame, label_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    
    # Draw threshold info
    cv2.putText(frame, f"Threshold: {THRESHOLD}  |  Press 'q' to quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    cv2.imshow("SignSpeak - Thay (ث) Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n👋 Detector closed.")