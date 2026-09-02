import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("="*60)
print("ALIF DETECTION")
print("="*60)

MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\alif_robust.tflite"

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Model loaded")
print(f"   Input shape: {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# MediaPipe Hands (using the correct import)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
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
print("Show your hand to test Alif detection")

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
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
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
            
            # Predict
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the score
            if prediction.shape[1] == 1:
                # Binary classification
                score = prediction[0][0]
                is_alif = score > 0.5
            else:
                # Multi-class - get max
                class_id = np.argmax(prediction[0])
                score = prediction[0][class_id]
                is_alif = class_id == 0  # Assuming class 0 is Alif
            
            # Display result
            if is_alif:
                text = f"✅ ALIF (ا) - {score*100:.1f}%"
                color = (0, 255, 0)
            else:
                text = f"❌ Not Alif - {score*100:.1f}%"
                color = (0, 0, 255)
            
            # Show on frame
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    cv2.imshow('Alif Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Done!")