# Save as: testing/debug_prediction.py

import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
import sqlite3
import os

print("="*60)
print("SIGNSPEAK - Prediction Debug Tool")
print("="*60)

# Paths
DB_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\dataset\main_dataset.db"
MODEL_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models\alphabet_model_final.h5"
ENCODER_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models\label_encoder_final.pkl"
SCALER_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models\scaler.pkl"

# ========== CHECK 1: HOW WAS DATA STORED? ==========
print("\n📊 CHECK 1: Database Feature Format")
print("-"*40)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get one sample
cursor.execute("SELECT * FROM rightHandDataset LIMIT 1")
sample = cursor.fetchone()
columns = [desc[0] for desc in cursor.description]

print(f"Columns: {columns}")
print(f"Sample data:")
for i, (col, val) in enumerate(zip(columns, sample)):
    if i < 5 or i == len(columns)-1:
        print(f"  {col}: {val}")
    elif i == 5:
        print(f"  ... (42 feature columns)")

# Check value ranges
cursor.execute("SELECT MIN(x1), MAX(x1), MIN(y1), MAX(y1) FROM rightHandDataset")
min_x, max_x, min_y, max_y = cursor.fetchone()
print(f"\nFeature ranges:")
print(f"  X: {min_x:.2f} to {max_x:.2f}")
print(f"  Y: {min_y:.2f} to {max_y:.2f}")

# Check if values are normalized (0-1) or pixel values
if min_x > 1:
    print(f"  ⚠️ Features are PIXEL VALUES (not normalized)")
    print(f"  → Need to divide by image dimensions during training")
else:
    print(f"  ✅ Features are NORMALIZED (0-1 range)")

conn.close()

# ========== CHECK 2: WHAT THE MODEL EXPECTS ==========
print("\n📁 CHECK 2: Model Input Format")
print("-"*40)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)
print(f"Classes: {list(le.classes_)}")

# ========== CHECK 3: LIVE WEBCAM FEATURES ==========
print("\n📷 CHECK 3: Webcam Feature Extraction")
print("-"*40)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("Show your hand clearly to the camera...")
print("Press SPACE to capture and analyze features")
print("Press Q to quit\n")

try:
    use_scaler = os.path.exists(SCALER_PATH)
    if use_scaler:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        cv2.putText(display, "Press SPACE to analyze", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Debug', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # SPACE
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks (MediaPipe gives 0-1 normalized values)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])
                    
                    raw_features = np.array(landmarks, dtype=np.float32)
                    
                    print("\n" + "="*40)
                    print("RAW FEATURES (from MediaPipe):")
                    print(f"  Range: {raw_features.min():.4f} to {raw_features.max():.4f}")
                    print(f"  Mean: {raw_features.mean():.4f}")
                    print(f"  First 6 values: {raw_features[:6]}")
                    
                    # Try different preprocessing methods
                    print("\n--- Test 1: Raw (0-1) ---")
                    input1 = raw_features.reshape(1, -1).astype(np.float32)
                    pred1 = model.predict(input1, verbose=0)[0]
                    top3_1 = np.argsort(pred1)[-3:][::-1]
                    for i, idx in enumerate(top3_1):
                        print(f"  {i+1}. {le.inverse_transform([idx])[0]}: {pred1[idx]*100:.1f}%")
                    
                    print("\n--- Test 2: Multiply by 100 ---")
                    input2 = (raw_features * 100).reshape(1, -1).astype(np.float32)
                    pred2 = model.predict(input2, verbose=0)[0]
                    top3_2 = np.argsort(pred2)[-3:][::-1]
                    for i, idx in enumerate(top3_2):
                        print(f"  {i+1}. {le.inverse_transform([idx])[0]}: {pred2[idx]*100:.1f}%")
                    
                    print("\n--- Test 3: Multiply by 300 (image size) ---")
                    input3 = (raw_features * 300).reshape(1, -1).astype(np.float32)
                    pred3 = model.predict(input3, verbose=0)[0]
                    top3_3 = np.argsort(pred3)[-3:][::-1]
                    for i, idx in enumerate(top3_3):
                        print(f"  {i+1}. {le.inverse_transform([idx])[0]}: {pred3[idx]*100:.1f}%")
                    
                    if use_scaler:
                        print("\n--- Test 4: With Scaler ---")
                        try:
                            input4 = scaler.transform(raw_features.reshape(1, -1)).astype(np.float32)
                            pred4 = model.predict(input4, verbose=0)[0]
                            top3_4 = np.argsort(pred4)[-3:][::-1]
                            for i, idx in enumerate(top3_4):
                                print(f"  {i+1}. {le.inverse_transform([idx])[0]}: {pred4[idx]*100:.1f}%")
                        except Exception as e:
                            print(f"  Error with scaler: {e}")
                    
                    print("\n--- Test 5: With Scaler + Multiply by 100 ---")
                    try:
                        input5 = (raw_features * 100).reshape(1, -1)
                        if use_scaler:
                            input5 = scaler.transform(input5)
                        input5 = input5.astype(np.float32)
                        pred5 = model.predict(input5, verbose=0)[0]
                        top3_5 = np.argsort(pred5)[-3:][::-1]
                        for i, idx in enumerate(top3_5):
                            print(f"  {i+1}. {le.inverse_transform([idx])[0]}: {pred5[idx]*100:.1f}%")
                    except Exception as e:
                        print(f"  Error: {e}")
                    
                    print("\n✅ WHICH TEST GAVE THE CORRECT PREDICTION?")
                    print("   Tell me which test number predicted correctly!")
                    
            else:
                print("⚠️ No hand detected! Show your hand more clearly.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)