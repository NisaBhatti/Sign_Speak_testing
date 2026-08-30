# Save as: testing/minimal_with_model.py

import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
import re

print("Loading model...")
TFLITE_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models\alphabet_model_final.tflite"
ENCODER_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models\label_encoder_final.pkl"
SCALER_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models\scaler.pkl"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("Model loaded. Starting camera...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Sign Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sign Detection', 800, 600)

print("READY! Make signs. Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Create display with results area
    display = frame.copy()
    
    # Process
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw green dots
            for lm in landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
            
            # Extract features
            raw = np.array([v for lm in landmarks.landmark for v in (lm.x, lm.y)], dtype=np.float32)
            
            # TRY ALL 3 METHODS
            results_text = []
            
            # Method 1: Scale match
            scale = 0.394 / max(raw.mean(), 0.001)
            f1 = scaler.transform((raw * scale).reshape(1, -1)).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], f1)
            interpreter.invoke()
            p1 = interpreter.get_tensor(output_details[0]['index'])[0]
            lab1 = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', 
                         le.inverse_transform([np.argmax(p1)])[0]).strip()
            conf1 = p1.max() * 100
            
            # Method 2: Raw + Scaler  
            f2 = scaler.transform(raw.reshape(1, -1)).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], f2)
            interpreter.invoke()
            p2 = interpreter.get_tensor(output_details[0]['index'])[0]
            lab2 = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '',
                         le.inverse_transform([np.argmax(p2)])[0]).strip()
            conf2 = p2.max() * 100
            
            # Method 3: /300 + Scaler
            f3 = scaler.transform((raw / 300.0).reshape(1, -1)).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], f3)
            interpreter.invoke()
            p3 = interpreter.get_tensor(output_details[0]['index'])[0]
            lab3 = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '',
                         le.inverse_transform([np.argmax(p3)])[0]).strip()
            conf3 = p3.max() * 100
            
            # SHOW RESULTS ON SCREEN
            y = 30
            cv2.putText(display, "METHOD 1 (Scale):", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
            cv2.putText(display, f"{lab1} {conf1:.0f}%", (180, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            y += 30
            cv2.putText(display, "METHOD 2 (Raw+Scaler):", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
            cv2.putText(display, f"{lab2} {conf2:.0f}%", (180, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            y += 30
            cv2.putText(display, "METHOD 3 (Div300+Scaler):", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
            cv2.putText(display, f"{lab3} {conf3:.0f}%", (180, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    else:
        cv2.putText(display, "Show hand", (w//2-60, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imshow('Sign Detection', display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done!")