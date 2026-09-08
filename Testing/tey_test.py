import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import json
import os
import time

print("="*60)
print("TEY (ط) SIGN DETECTION - EXCLUSIVE TESTING")
print("="*60)

# ==================== CONFIGURATION ====================
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\tey_model_exclusive.tflite"
LABEL_ENCODER_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\tey_label_encoder_exclusive.pkl"
CONFIG_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\tey_config_exclusive.json"

# Load config to get threshold
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    CONFIDENCE_THRESHOLD = config.get('threshold', 0.7)
    print(f"✅ Loaded threshold: {CONFIDENCE_THRESHOLD}")
except:
    CONFIDENCE_THRESHOLD = 0.7
    print(f"⚠️ Using default threshold: {CONFIDENCE_THRESHOLD}")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    exit()

# ==================== LOAD MODEL ====================
print(f"\n📂 Loading model from: {MODEL_PATH}")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load label encoder
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅ Label encoder loaded: {label_encoder.classes_}")
except:
    label_encoder = None
    print(f"⚠️ Using default labels")

# ==================== SETUP MEDIAPIPE ====================
print(f"\n🔄 Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("✅ MediaPipe initialized")

# ==================== DETECTION FUNCTIONS ====================
def extract_hand_landmarks_from_frame(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None, results
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
        
        if len(landmarks) >= 42:
            wrist_x, wrist_y = landmarks[0], landmarks[1]
            for i in range(0, len(landmarks), 2):
                landmarks[i] -= wrist_x
                landmarks[i+1] -= wrist_y
            return np.array(landmarks, dtype=np.float32), results
        
        return None, results
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def predict_tey(landmarks):
    try:
        if landmarks.shape[0] != 42:
            landmarks = landmarks[:42] if len(landmarks) > 42 else np.pad(landmarks, (0, 42 - len(landmarks)))
        
        features = landmarks.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        probability = float(prediction[0][0])
        
        # ONLY detect if probability is above threshold
        is_tey = probability > CONFIDENCE_THRESHOLD
        
        if label_encoder:
            class_id = 1 if is_tey else 0
            class_name = label_encoder.inverse_transform([class_id])[0]
        else:
            class_name = 'tey' if is_tey else 'not_tey'
        
        return class_name, probability, is_tey
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 'error', 0.0, False

# ==================== CAMERA SETUP ====================
print(f"\n📷 Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"✅ Camera opened successfully!")
print(f"\n" + "="*60)
print("INSTRUCTIONS:")
print(f"   ✋ Show the Tey (ط) hand sign to the camera")
print(f"   🟢 Green box = TEY detected (confidence > {CONFIDENCE_THRESHOLD})")
print(f"   🔴 Red box = Not Tey")
print(f"   ⚠️ Alif and other signs will be REJECTED")
print("\nControls:")
print("   'q' - Quit")
print("   's' - Save screenshot")
print("   't' - Toggle confidence threshold")
print("="*60 + "\n")

# ==================== DETECTION LOOP ====================
frame_count = 0
screenshot_count = 0
detection_history = []
current_threshold = CONFIDENCE_THRESHOLD

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame. Retrying...")
        continue
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    landmarks, results = extract_hand_landmarks_from_frame(frame)
    
    if landmarks is not None and results and results.multi_hand_landmarks:
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Predict
        class_name, probability, is_tey = predict_tey(landmarks)
        
        # Prepare display
        if is_tey:
            text = f"✅ TEY (ط) DETECTED"
            color = (0, 255, 0)
            # Draw green border
            cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 4)
            cv2.rectangle(frame, (5, 5), (w-5, h-5), (0, 255, 0), 2)
            cv2.putText(frame, "⭐ TEY", (w-120, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            text = f"❌ Rejected (Not Tey)"
            color = (0, 0, 255)
            # Draw red border
            cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 0, 255), 2)
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Confidence bar
        bar_width = 300
        bar_height = 30
        bar_x = 20
        bar_y = 80
        confidence_width = int(bar_width * probability)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Color based on threshold
        if probability > current_threshold:
            color_bar = (0, 255, 0)
        else:
            color_bar = (0, 0, 255)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color_bar, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Threshold line
        threshold_x = bar_x + int(bar_width * current_threshold)
        cv2.line(frame, (threshold_x, bar_y-5), (threshold_x, bar_y+bar_height+5), (255, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {current_threshold:.2f}", (threshold_x-30, bar_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Confidence percentage
        cv2.putText(frame, f"{probability*100:.1f}%", (bar_x + bar_width + 20, bar_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
    else:
        # No hand detected
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(frame, "🤚 Show TEY (ط) sign ONLY", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(frame, "Other signs will be rejected", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Instructions
    cv2.putText(frame, "q=quit | s=screenshot | t=threshold", (10, frame.shape[0]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Tey (ط) Exclusive Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"tey_detection_{timestamp}.png", frame)
        print(f"📸 Screenshot saved")
    elif key == ord('t'):
        # Toggle threshold
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        idx = thresholds.index(current_threshold) if current_threshold in thresholds else 0
        current_threshold = thresholds[(idx + 1) % len(thresholds)]
        print(f"📊 Threshold set to: {current_threshold:.2f}")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "="*60)
print("✅ DETECTION COMPLETED")
print("="*60)
print(f"   Screenshots saved: {screenshot_count}")
print("\n👋 Goodbye!")
print("="*60)