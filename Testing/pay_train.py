# Save as: pay_train.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
import os
import json
import sqlite3
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SIGNSPEAK - PAY (پے) DETECTION TRAINING")
print("="*60)

# ========== PATHS ==========
DATASET_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Pay"
DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "pay_model.tflite")
OUTPUT_INFO = os.path.join(OUTPUT_DIR, "pay_model_info.json")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📁 Dataset Path: {DATASET_PATH}")
print(f"📁 Output Path: {OUTPUT_DIR}")

# ========== CHECK DATASET ==========
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset path not found: {DATASET_PATH}")
    exit()

# Get all image files
image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
    image_files.extend([f for f in os.listdir(DATASET_PATH) if f.lower().endswith(ext)])

print(f"\n📊 Found {len(image_files)} images in Pay folder")

if len(image_files) == 0:
    print("\n❌ No images found in the Pay folder!")
    exit()

# ========== INITIALIZE MEDIAPIPE ==========
print("\n🖐️ Initializing MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ========== EXTRACT LANDMARKS ==========
def extract_landmarks(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y])
        
        return np.array(landmarks, dtype=np.float32)
    except:
        return None

# ========== EXTRACT PAY SAMPLES ==========
print("\n" + "="*60)
print("📊 EXTRACTING PAY SAMPLES")
print("="*60)

pay_samples = []
for idx, filename in enumerate(image_files):
    filepath = os.path.join(DATASET_PATH, filename)
    landmarks = extract_landmarks(filepath)
    if landmarks is not None:
        pay_samples.append(landmarks)
    if (idx + 1) % 10 == 0:
        print(f"   Processed {idx + 1}/{len(image_files)} - Found {len(pay_samples)}")

print(f"\n✅ Extracted {len(pay_samples)} Pay samples")

if len(pay_samples) == 0:
    print("\n❌ No valid Pay samples found!")
    exit()

pay_samples = np.array(pay_samples, dtype=np.float32)

# ========== GET NON-PAY SAMPLES ==========
print("\n" + "="*60)
print("📊 GETTING NON-PAY SAMPLES")
print("="*60)

non_pay_samples = []

if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rightHandDataset")
        all_data = cursor.fetchall()
        conn.close()
        
        for row in all_data:
            label = str(row[43]).strip()
            if label and label != 'پے':
                features = [float(row[i]) for i in range(1, 43)]
                non_pay_samples.append(np.array(features, dtype=np.float32))
        
        print(f"   Found {len(non_pay_samples)} non-Pay samples from database")
    except Exception as e:
        print(f"   Error loading database: {e}")

# Generate synthetic if needed
if len(non_pay_samples) < len(pay_samples):
    needed = len(pay_samples) * 2 - len(non_pay_samples)
    print(f"   Generating {needed} synthetic samples...")
    for _ in range(needed):
        sample = np.random.rand(42).astype(np.float32)
        non_pay_samples.append(sample)

non_pay_samples = np.array(non_pay_samples, dtype=np.float32)

# ========== BALANCE DATASET ==========
print("\n" + "="*60)
print("📊 BALANCING DATASET")
print("="*60)

num_pay = len(pay_samples)
num_non_pay = len(non_pay_samples)

print(f"   Pay: {num_pay}, Non-Pay: {num_non_pay}")

# Balance
if num_non_pay > num_pay * 2:
    indices = np.random.choice(num_non_pay, num_pay * 2, replace=False)
    non_pay_samples = non_pay_samples[indices]
else:
    # Duplicate non-pay samples to match pay
    while len(non_pay_samples) < num_pay:
        extra = non_pay_samples[:min(num_pay - len(non_pay_samples), len(non_pay_samples))]
        non_pay_samples = np.vstack([non_pay_samples, extra])

num_non_pay = len(non_pay_samples)
print(f"   After balancing - Pay: {num_pay}, Non-Pay: {num_non_pay}")

# Create dataset
X = np.vstack([pay_samples, non_pay_samples])
y = np.hstack([np.ones(num_pay), np.zeros(num_non_pay)])

print(f"   Total: {len(X)} samples")

# ========== SPLIT DATA ==========
print("\n" + "="*60)
print("📊 SPLITTING DATA")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# ========== BUILD SIMPLE MODEL ==========
print("\n" + "="*60)
print("🧠 BUILDING MODEL")
print("="*60)

def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(42,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model()
model.summary()

# ========== TRAIN ==========
print("\n" + "="*60)
print("🏋️ TRAINING MODEL")
print("="*60)

# Simple callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ========== EVALUATE ==========
print("\n" + "="*60)
print("📊 EVALUATING")
print("="*60)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n   Accuracy: {accuracy*100:.2f}%")
print(f"\n   Confusion Matrix:")
print(f"   [[{conf_matrix[0,0]} {conf_matrix[0,1]}]")
print(f"    [{conf_matrix[1,0]} {conf_matrix[1,1]}]]")

# ========== SAVE TFLITE ==========
print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(OUTPUT_MODEL, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved: {OUTPUT_MODEL}")
    print(f"   Size: {os.path.getsize(OUTPUT_MODEL) / 1024:.2f} KB")
except Exception as e:
    print(f"⚠️ TFLite conversion error: {e}")
    model.save(os.path.join(OUTPUT_DIR, 'pay_model.h5'))
    print(f"✅ H5 model saved")

# ========== SAVE INFO ==========
info = {
    "model": "Pay Detection Model",
    "accuracy": float(accuracy),
    "threshold": 0.5,
    "pay_samples": int(num_pay),
    "non_pay_samples": int(num_non_pay),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test))
}

with open(OUTPUT_INFO, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2)

print(f"✅ Info saved: {OUTPUT_INFO}")

# ========== PLOT ==========
try:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pay_training.png'), dpi=150)
    plt.show()
    print(f"✅ Plot saved")
except:
    print("⚠️ Could not save plot")

# ========== DONE ==========
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\n📁 Output: {OUTPUT_DIR}")
print(f"   Model: {OUTPUT_MODEL}")
print(f"   Info: {OUTPUT_INFO}")
print(f"\n📊 Accuracy: {accuracy*100:.2f}%")
print("\n🔍 To test: python test_pay.py")

hands.close()
print("\n✅ Done!")