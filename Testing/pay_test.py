# Save as: test_pay.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import json

print("="*60)
print("PAY DETECTION TEST (FIXED)")
print("="*60)

# Paths
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\pay_model.tflite"

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit()

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Increased confidence
    min_tracking_confidence=0.5
)

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n📷 Camera opened! Press 'q' to quit")
print("Show 'Pay' hand sign (palm facing camera with fingers together)")

def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks for consistent input to the model
    """
    # Convert to numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Center around wrist (landmark 0)
    wrist = points[0]
    centered = points - wrist
    
    # Normalize scale using distance from wrist to middle finger MCP (landmark 9)
    scale = np.linalg.norm(points[9] - wrist)
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    # Flatten to 1D array (21 landmarks * 3 coordinates = 63 features)
    return normalized.flatten()

def normalize_landmarks_2d(landmarks):
    """
    Alternative: 2D normalization (x, y only) - use if your model expects 42 features
    """
    points = np.array([[lm.x, lm.y] for lm in landmarks])
    wrist = points[0]
    centered = points - wrist
    scale = np.linalg.norm(points[9] - wrist)
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    return normalized.flatten()

print("\n" + "="*60)
print("TIPS FOR ACCURATE DETECTION:")
print("1. Show palm facing the camera")
print("2. Keep fingers together (like a 'pay' gesture)")
print("3. Ensure good lighting")
print("4. Position hand in center of frame")
print("="*60 + "\n")

# Variables for smoothing predictions
prediction_history = []
SMOOTHING_WINDOW = 5

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and normalize features
            # Option 1: 3D normalization (63 features)
            features_3d = normalize_landmarks(hand_landmarks.landmark)
            
            # Option 2: 2D normalization (42 features) - uncomment if needed
            # features_2d = normalize_landmarks_2d(hand_landmarks.landmark)
            
            # Check model input shape and use appropriate features
            expected_features = input_details[0]['shape'][1]
            
            if expected_features == 63:
                features = features_3d
            elif expected_features == 42:
                features = normalize_landmarks_2d(hand_landmarks.landmark)
            else:
                print(f"⚠️ Unexpected feature size: {expected_features}")
                features = features_3d
            
            features = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Predict
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            # Smooth predictions
            prediction_history.append(prediction)
            if len(prediction_history) > SMOOTHING_WINDOW:
                prediction_history.pop(0)
            smoothed_prediction = np.mean(prediction_history)
            
            # Show result with threshold adjustment
            threshold = 0.6  # Adjust this threshold
            is_pay = smoothed_prediction > threshold
            
            # Create info panel
            info_panel = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)
            
            if is_pay:
                text = f"✅ PAY DETECTED - {smoothed_prediction*100:.1f}%"
                color = (0, 255, 0)
                cv2.rectangle(info_panel, (10, 10), (frame.shape[1]-10, 110), (0, 255, 0), 2)
            else:
                text = f"❌ Not Pay - {smoothed_prediction*100:.1f}%"
                color = (0, 0, 255)
                cv2.rectangle(info_panel, (10, 10), (frame.shape[1]-10, 110), (0, 0, 255), 2)
            
            cv2.putText(info_panel, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Show confidence bar
            bar_width = int((frame.shape[1] - 40) * smoothed_prediction)
            cv2.rectangle(info_panel, (20, 70), (20 + bar_width, 90), (255, 255, 255), -1)
            cv2.putText(info_panel, f"Confidence: {smoothed_prediction*100:.1f}%", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Combine frames
            frame = np.vstack([info_panel, frame])
            
    else:
        # No hand detected
        info_panel = np.zeros((80, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(info_panel, "🤚 Show your hand", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        frame = np.vstack([info_panel, frame])
    
    cv2.putText(frame, "Q=Quit", (frame.shape[1]-100, frame.shape[0]-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    cv2.imshow('Pay Detection (Fixed)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Done!")