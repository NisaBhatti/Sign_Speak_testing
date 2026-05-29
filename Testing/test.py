# Save as: MODEL/Testing/test_fixed.py

import cv2
import numpy as np
import tensorflow as tf
import os
import time
from collections import Counter

print("🧪 Urdu Alphabet Recognition - Fixed Version")
print("="*50)

# Paths
model_path = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\urdu_alphabets_full.tflite"
labels_path = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\labels.txt"
reference_images_path = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset"

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("Please train the model first using train_simple_folders.py")
    exit()

# Load TFLite model
print("📱 Loading TensorFlow Lite model...")
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load labels
try:
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(labels)} labels")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    exit()

# Load reference images (only first few to save memory)
print("📚 Loading reference images...")
reference_images = {}
loaded_count = 0

for alphabet in labels[:20]:  # Load first 20 to save memory
    ref_folder = os.path.join(reference_images_path, alphabet)
    if os.path.exists(ref_folder):
        images = [f for f in os.listdir(ref_folder) if f.endswith('.png')]
        if images:
            ref_img_path = os.path.join(ref_folder, images[0])
            ref_img = cv2.imread(ref_img_path)
            if ref_img is not None:
                ref_img = cv2.resize(ref_img, (150, 150))
                reference_images[alphabet] = ref_img
                loaded_count += 1

print(f"✅ Loaded {loaded_count} reference images")

# Initialize webcam with error handling
print("\n🎥 Starting webcam...")

# Try different camera indices
camera_index = 0
cap = None

for idx in range(3):
    print(f"Trying camera index {idx}...")
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print(f"✅ Camera {idx} opened successfully!")
        camera_index = idx
        break
    else:
        print(f"❌ Could not open camera {idx}")

if cap is None or not cap.isOpened():
    print("❌ Could not open any camera!")
    exit()

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Test if we can read frames
ret, test_frame = cap.read()
if not ret:
    print("❌ Camera opened but cannot read frames!")
    cap.release()
    exit()

print("✅ Webcam ready!")
print("\n📖 Instructions:")
print("  - Show Urdu alphabet sign to camera")
print("  - Press 'q' to quit")
print("  - Press 's' to save screenshot")
print("\n🎯 Starting prediction...")

# Variables
prediction_history = []
history_length = 5
frame_counter = 0
last_predictions = []
fps = 0
fps_counter = 0
fps_time = time.time()

# Create window
cv2.namedWindow('Urdu Alphabet Recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Urdu Alphabet Recognition', 1000, 500)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue
        
        frame_counter += 1
        frame_display = frame.copy()
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
        
        # Every few frames do prediction
        if frame_counter % 3 == 0:
            # Preprocess
            img = cv2.resize(frame, (128, 128))
            img_input = np.expand_dims(img, axis=0)
            img_input = img_input.astype(np.float32) / 255.0
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            
            # Get top prediction
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx] * 100
            current_prediction = labels[pred_idx]
            
            # Store in history
            last_predictions.append(current_prediction)
            if len(last_predictions) > history_length:
                last_predictions.pop(0)
            
            # Get smoothed prediction
            if last_predictions:
                most_common = Counter(last_predictions).most_common(1)[0]
                smoothed_prediction = most_common[0]
                # Only use smoothed if confidence is high
                if confidence > 60:
                    display_prediction = smoothed_prediction
                else:
                    display_prediction = current_prediction
            else:
                display_prediction = current_prediction
            
            # Get top 3 for display
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            top_3_labels = [labels[i] for i in top_3_idx]
            top_3_scores = [predictions[0][i] * 100 for i in top_3_idx]
        
        # Display info on frame - Left side
        y = 30
        
        # FPS
        cv2.putText(frame_display, f"FPS: {fps}", (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        # Main prediction
        cv2.putText(frame_display, f"PREDICTION: {display_prediction}", (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
        
        # Confidence
        cv2.putText(frame_display, f"Confidence: {confidence:.1f}%", (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
        
        # Top 3 predictions
        cv2.putText(frame_display, "Top 3:", (10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        for i, (label, score) in enumerate(zip(top_3_labels, top_3_scores)):
            color = (0, 255, 0) if i == 0 else (150, 150, 150)
            cv2.putText(frame_display, f"{i+1}. {label}: {score:.1f}%", (20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 22
        
        # Instructions
        cv2.putText(frame_display, "Press 'q' to quit | 's' to save", (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Create reference panel - Right side
        ref_panel = np.ones((frame.shape[0], 250, 3), dtype=np.uint8) * 240
        
        # Title
        cv2.putText(ref_panel, "REFERENCE", (60, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Show reference image
        if display_prediction in reference_images:
            ref_img = reference_images[display_prediction]
            # Place reference image
            x_offset = (250 - ref_img.shape[1]) // 2
            ref_panel[60:60+ref_img.shape[0], x_offset:x_offset+ref_img.shape[1]] = ref_img
            
            # Label
            cv2.putText(ref_panel, display_prediction, (80, 280), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Match indicator
            if confidence > 80:
                status = "✓ GOOD MATCH"
                color = (0, 255, 0)
            elif confidence > 60:
                status = "? MAYBE"
                color = (0, 165, 255)
            else:
                status = "✗ UNCLEAR"
                color = (0, 0, 255)
            
            cv2.putText(ref_panel, status, (65, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Confidence bar
            cv2.rectangle(ref_panel, (30, 350), (220, 370), (200, 200, 200), -1)
            bar_width = int((confidence / 100) * 190)
            if confidence > 80:
                bar_color = (0, 255, 0)
            elif confidence > 60:
                bar_color = (0, 165, 255)
            else:
                bar_color = (0, 0, 255)
            cv2.rectangle(ref_panel, (30, 350), (30 + bar_width, 370), bar_color, -1)
            cv2.putText(ref_panel, f"{confidence:.0f}%", (230, 367), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.putText(ref_panel, "No reference", (60, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Combine frames
        combined = np.hstack((frame_display, ref_panel))
        
        # Show
        cv2.imshow('Urdu Alphabet Recognition', combined)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("saved_predictions", exist_ok=True)
            save_path = f"saved_predictions/{display_prediction}_{timestamp}.jpg"
            cv2.imwrite(save_path, combined)
            print(f"💾 Saved: {save_path}")
    
    except Exception as e:
        print(f"⚠️ Error in main loop: {e}")
        continue

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Testing completed!")