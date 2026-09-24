# Save as: D:\MODEL\Sign_Speak_testing-main\model_training\train_thay.py

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import glob
import random

print("="*60)
print("SIGNSPEAK - Training Thay (ث) DETECTOR")
print("="*60)

# ========== CONFIGURATION ==========
DATASET_ROOT = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset"
THAY_FOLDER = os.path.join(DATASET_ROOT, "Thay")
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\thay_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

# Arabic letter for Thay
THAY_LETTER = 'ث'

# ========== MEDIAPIPE SETUP ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

def extract_landmarks(image_path):
    """Extract 42 normalized landmarks (21 points × x,y) from image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    hand = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand.landmark:
        landmarks.append(lm.x)  # x already normalized 0-1
        landmarks.append(lm.y)  # y already normalized 0-1
    
    return np.array(landmarks, dtype=np.float32)


# ========== LOAD THAY (POSITIVE) SAMPLES ==========
print(f"\n📁 Loading Thay (ث) images from: {THAY_FOLDER}")

if not os.path.exists(THAY_FOLDER):
    raise FileNotFoundError(f"❌ Folder not found: {THAY_FOLDER}")

thay_images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    thay_images.extend(glob.glob(os.path.join(THAY_FOLDER, ext)))

print(f"   Found {len(thay_images)} Thay images")

X_thay = []
failed_thay = 0

for i, img_path in enumerate(thay_images):
    landmarks = extract_landmarks(img_path)
    if landmarks is not None:
        X_thay.append(landmarks)
    else:
        failed_thay += 1
    if (i + 1) % 50 == 0:
        print(f"   Processed {i+1}/{len(thay_images)}...")

print(f"   ✅ Extracted: {len(X_thay)} | ❌ Failed (no hand): {failed_thay}")

if len(X_thay) < 5:
    raise ValueError("❌ Not enough Thay samples! Need at least 5 valid hand images.")

X_thay = np.array(X_thay, dtype=np.float32)


# ========== LOAD NEGATIVE SAMPLES (OTHER LETTERS) ==========
print(f"\n📁 Loading negative samples (other letters)...")

# Get all subfolders except "Thay"
other_folders = [
    os.path.join(DATASET_ROOT, d) 
    for d in os.listdir(DATASET_ROOT) 
    if os.path.isdir(os.path.join(DATASET_ROOT, d)) 
    and d.lower() != "thay"
]

print(f"   Found {len(other_folders)} other letter folders")

X_other = []
max_other_per_folder = 200  # Cap per folder to keep balance

for folder in other_folders:
    folder_name = os.path.basename(folder)
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        images.extend(glob.glob(os.path.join(folder, ext)))
    
    if not images:
        continue
    
    # Random sample if too many
    if len(images) > max_other_per_folder:
        images = random.sample(images, max_other_per_folder)
    
    extracted = 0
    for img_path in images:
        landmarks = extract_landmarks(img_path)
        if landmarks is not None:
            X_other.append(landmarks)
            extracted += 1
    
    print(f"   {folder_name}: {extracted} samples")

X_other = np.array(X_other, dtype=np.float32) if X_other else np.zeros((0, 42), dtype=np.float32)
print(f"   ✅ Total negative samples: {len(X_other)}")


# ========== AUGMENT THAY DATA (make it robust) ==========
print(f"\n🔧 Augmenting Thay data for robustness...")
print(f"   Original Thay samples: {len(X_thay)}")

augmented_X = [X_thay]

# Augmentation 1: Small noise (simulates camera noise)
for _ in range(2):
    noise = np.random.normal(0, 0.015, X_thay.shape).astype(np.float32)
    X_noisy = np.clip(X_thay + noise, 0, 1)
    augmented_X.append(X_noisy)

# Augmentation 2: Scale variations (different distances)
for scale in [0.92, 1.08]:
    X_scaled = np.clip(X_thay * scale, 0, 1)
    augmented_X.append(X_scaled)

# Augmentation 3: Shift variations (different positions)
for shift_x, shift_y in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02), (0.015, 0.015), (-0.015, -0.015)]:
    X_shifted = X_thay.copy()
    X_shifted[:, 0::2] = np.clip(X_shifted[:, 0::2] + shift_x, 0, 1)
    X_shifted[:, 1::2] = np.clip(X_shifted[:, 1::2] + shift_y, 0, 1)
    augmented_X.append(X_shifted)

# Augmentation 4: Slight rotation (small angle)
def rotate_landmarks(landmarks, angle_deg):
    """Rotate landmarks around center (0.5, 0.5)."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = landmarks.copy()
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i] - 0.5, landmarks[i+1] - 0.5
        rotated[i] = x * cos_a - y * sin_a + 0.5
        rotated[i+1] = x * sin_a + y * cos_a + 0.5
    return np.clip(rotated, 0, 1)

for angle in [-8, 8]:
    X_rotated = np.array([rotate_landmarks(lm, angle) for lm in X_thay], dtype=np.float32)
    augmented_X.append(X_rotated)

X_thay_aug = np.vstack(augmented_X)
print(f"   Augmented Thay samples: {len(X_thay_aug)}")


# ========== BUILD BALANCED DATASET ==========
print(f"\n📊 Creating balanced dataset...")

n_thay = len(X_thay_aug)
n_other = min(len(X_other), n_thay * 2)  # 2x negatives for better specificity

if n_other == 0:
    raise ValueError("❌ No negative samples found! Add other letter folders.")

np.random.seed(42)
indices = np.random.choice(len(X_other), n_other, replace=False)
X_other_bal = X_other[indices]

X = np.vstack([X_thay_aug, X_other_bal])
y = np.hstack([np.ones(n_thay), np.zeros(n_other)])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} samples")
print(f"   Thay (1): {int(y.sum())} | Other (0): {len(y) - int(y.sum())}")


# ========== SPLIT ==========
print(f"\n✂️ Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ========== CLASS WEIGHTS ==========
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\n⚖️ Class weights: {class_weight_dict}")


# ========== BUILD MODEL ==========
print(f"\n🏗️ Building model...")

model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()


# ========== TRAIN ==========
print(f"\n🚀 Training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=1),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_thay.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)


# ========== EVALUATE ==========
print(f"\n📊 Final Results:")
train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"   Train: Acc={train_acc*100:.2f}%, AUC={train_auc:.4f}")
print(f"   Val:   Acc={val_acc*100:.2f}%, AUC={val_auc:.4f}")
print(f"   Test:  Acc={test_acc*100:.2f}%, AUC={test_auc:.4f}")

# Test on ORIGINAL (non-augmented) Thay samples
thay_preds = model.predict(X_thay, verbose=0).flatten()
thay_detected = (thay_preds > 0.5).sum()
thay_rate = thay_detected / len(X_thay) * 100
print(f"\n   ✅ Original Thay detection: {thay_rate:.1f}% ({thay_detected}/{len(X_thay)})")
print(f"   Avg confidence: {thay_preds.mean():.4f}")
print(f"   Min confidence: {thay_preds.min():.4f}")

# Test on negative samples
if len(X_other) > 0:
    other_preds = model.predict(X_other, verbose=0).flatten()
    false_positives = (other_preds > 0.5).sum()
    specificity = (1 - false_positives / len(X_other)) * 100
    print(f"\n   ❌ False positives: {false_positives}/{len(X_other)} ({specificity:.1f}% specificity)")


# ========== SAVE ==========
print(f"\n💾 Saving models...")

model.save(os.path.join(OUTPUT_DIR, 'thay_detector.h5'))

# TFLite for deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(OUTPUT_DIR, 'thay_detector.tflite'), 'wb') as f:
    f.write(tflite_model)

print(f"   TFLite size: {len(tflite_model)/1024:.1f} KB")

# Info JSON
info = {
    'model': 'Thay Detector',
    'letter': THAY_LETTER,
    'letter_name': 'Thay',
    'input': '42 MediaPipe landmarks (x,y normalized 0-1)',
    'preprocessing': 'Raw MediaPipe values, no scaler',
    'augmentation': 'Noise, Scale, Shift, Rotation',
    'num_original_thay_samples': int(len(X_thay)),
    'num_augmented_thay_samples': int(len(X_thay_aug)),
    'num_negative_samples': int(len(X_other)),
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'thay_detection_rate': float(thay_rate),
    'threshold': 0.5,
    'classes': ['Not Thay', 'Thay']
}

with open(os.path.join(OUTPUT_DIR, 'thay_detector_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ TRAINING COMPLETE!")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Thay Detection Rate: {thay_rate:.1f}%")
print(f"   Model saved to: {OUTPUT_DIR}")