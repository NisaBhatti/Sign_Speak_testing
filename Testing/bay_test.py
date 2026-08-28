# Save as: test2_bay_landmark.py in Testing folder

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time
from collections import Counter

print("🧪 Bay (ب) Detection - Landmark Based")
print("="*50)

# Paths
model_path = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_bay\bay_robust.tflite"

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    exit()

# Load TFLite model
print("📱 Loading TensorFlow Lite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize MediaPipe Hands
print("🖐️ Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe initialized!")

# Initialize webcam
print("\n🎥 Starting webcam...")

# Try different camera indices
cap = None
for idx in range(3):
    print(f"Trying camera index {idx}...")
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print(f"✅ Camera {idx} opened successfully!")
        break
    else:
        print(f"❌ Could not open camera {idx}")

if cap is None or not cap.isOpened():
    print("❌ Could not open any camera!")
    exit()

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam ready!")
print("\n📖 Instructions:")
print("  - Show Bay (ب) hand sign to camera")
print("  - Press 'q' to quit")
print("  - Press 's' to save screenshot")
print("\n🎯 Starting Bay detection...")

# Variables for smoothing
prediction_history = []
history_length = 10  # Increased for better smoothing
fps = 0
fps_counter = 0
fps_time = time.time()
display_prediction = "Waiting..."
confidence = 0.0
last_valid_prediction = "Unknown"

# Landmark connections for visualization
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Reset detection flag
        hand_detected = False
        
        # Check if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_detected = True
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame_display,
                    hand_landmarks,
                    HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract features - 21 landmarks x 2 (x, y) = 42 features
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y])
                
                # Convert to numpy array with correct shape [1, 42]
                features = np.array(features, dtype=np.float32).reshape(1, -1)
                
                # Make sure we have exactly 42 features
                if features.shape[1] != 42:
                    print(f"Warning: Expected 42 features, got {features.shape[1]}")
                    continue
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], features)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                
                # Get the score (binary classification)
                if prediction.shape[1] == 1:
                    confidence = prediction[0][0] * 100
                    is_bay = confidence > 50
                else:
                    class_id = np.argmax(prediction[0])
                    confidence = prediction[0][class_id] * 100
                    is_bay = class_id == 0  # Assuming class 0 is Bay
                
                # Smooth predictions
                current_pred = "Bay (ب)" if is_bay else "Not Bay"
                prediction_history.append(is_bay)
                if len(prediction_history) > history_length:
                    prediction_history.pop(0)
                
                # Get smoothed prediction
                if len(prediction_history) >= 3:  # Need at least 3 predictions for smoothing
                    bay_count = sum(prediction_history)
                    smoothed_is_bay = bay_count > len(prediction_history) / 2
                    smoothed_pred = "Bay (ب)" if smoothed_is_bay else "Not Bay"
                    
                    # Use smoothed if confidence is moderate
                    if confidence > 55:
                        display_prediction = smoothed_pred
                        if smoothed_is_bay:
                            last_valid_prediction = "Bay (ب)"
                        else:
                            last_valid_prediction = "Not Bay"
                    else:
                        display_prediction = current_pred
                        if is_bay:
                            last_valid_prediction = "Bay (ب)"
                        else:
                            last_valid_prediction = "Not Bay"
                else:
                    display_prediction = current_pred
                    if is_bay:
                        last_valid_prediction = "Bay (ب)"
                    else:
                        last_valid_prediction = "Not Bay"
        
        # Display results on frame
        y = 30
        
        # FPS
        cv2.putText(frame_display, f"FPS: {fps}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 30
        
        # Main prediction
        if hand_detected:
            if display_prediction == "Bay (ب)":
                text = f"✅ BAY (ب) DETECTED!"
                color = (0, 255, 0)
            else:
                text = f"❌ NOT BAY"
                color = (0, 0, 255)
            
            cv2.putText(frame_display, text, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            y += 40
            
            # Confidence
            cv2.putText(frame_display, f"Confidence: {confidence:.1f}%", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 35
            
            # Confidence bar
            bar_x, bar_y = 10, y
            bar_w, bar_h = 300, 25
            cv2.rectangle(frame_display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         (50, 50, 50), -1)
            bar_width = int((confidence / 100) * bar_w)
            if confidence > 80:
                bar_color = (0, 255, 0)
            elif confidence > 60:
                bar_color = (0, 255, 255)
            else:
                bar_color = (0, 0, 255)
            cv2.rectangle(frame_display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h), 
                         bar_color, -1)
            cv2.putText(frame_display, f"{confidence:.0f}%", (bar_x + bar_w + 10, bar_y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 45
            
            # Show prediction label
            cv2.putText(frame_display, f"Prediction: {display_prediction}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 30
            
            # Draw wrist analysis (highlight wrist landmarks)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Landmark 0 = Wrist
                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    cv2.circle(frame_display, (wrist_x, wrist_y), 12, (0, 255, 255), -1)
                    cv2.putText(frame_display, "WRIST", (wrist_x - 30, wrist_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Draw fingertips (landmarks 4, 8, 12, 16, 20)
                    fingertips = [4, 8, 12, 16, 20]
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    for i, tip in enumerate(fingertips):
                        lm = hand_landmarks.landmark[tip]
                        tip_x, tip_y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_display, (tip_x, tip_y), 8, (0, 255, 0), -1)
                        cv2.putText(frame_display, finger_names[i][0], (tip_x - 10, tip_y - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        else:
            # No hand detected
            cv2.rectangle(frame_display, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
            cv2.putText(frame_display, "✋ NO HAND DETECTED", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame_display, "Show your hand to camera", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Reset display prediction when no hand
            display_prediction = "No Hand"
        
        # Instructions at bottom
        cv2.putText(frame_display, "Press 'q' to quit | 's' to save", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show the frame
        cv2.imshow('Bay (ب) Hand Sign Detection', frame_display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("saved_bay_predictions", exist_ok=True)
            save_path = f"saved_bay_predictions/bay_{timestamp}.jpg"
            cv2.imwrite(save_path, frame_display)
            print(f"💾 Saved: {save_path}")
    
    except Exception as e:
        print(f"⚠️ Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        continue

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Bay testing completed!")