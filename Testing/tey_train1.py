# Save as: model_training/train_tey_robust.py
"""
SignSpeak - Train Tey (ٹ) detector.

Tey (ٹ) = closed fist with thumb extended UPWARD (thumbs-up style).
This script trains a BINARY classifier: 1 = Tey, 0 = Not Tey.
"""

import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import os
import re
import glob
import cv2
import mediapipe as mp

print("=" * 60)
print("SIGNSPEAK - Training Tey (ٹ) ROBUST Model")
print("=" * 60)

# ========== CONFIGURATION ==========
DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
Tey_IMAGE_DIR = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Tey"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_tey"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

# Arabic letter for Tey
TEY_LETTER = 'ٹ'

# ========== MEDIAPIPE SETUP (for image-based extraction) ==========
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def extract_landmarks_from_image(img_path):
    """Extract 42 features (21 landmarks * x,y) from an image using MediaPipe.
    Returns normalized features in 0-1 range, or None if no hand detected."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    features = []
    for lm in hand.landmark:
        features.append(lm.x)  # already normalized 0-1
        features.append(lm.y)
    return features  # 42 values


# ========== LOAD DATA FROM DATABASE ==========
print("\n📁 Loading data from database...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM rightHandDataset")
all_data = cursor.fetchall()
conn.close()

print(f"   Total DB samples: {len(all_data)}")

X_db = []
y_db = []

for row in all_data:
    features = [float(row[i]) for i in range(1, 43)]
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    X_db.append(features)
    y_db.append(label)

X_db = np.array(X_db, dtype=np.float32)
y_db = np.array(y_db)

# ========== NORMALIZE DB DATA (pixel → 0-1) ==========
print(f"\n📏 Normalizing DB features...")
print(f"   Before: min={X_db.min():.1f}, max={X_db.max():.1f}")

X_db_norm = X_db.copy()
X_db_norm[:, 0::2] = X_db[:, 0::2] / IMAGE_WIDTH   # X coords
X_db_norm[:, 1::2] = X_db[:, 1::2] / IMAGE_HEIGHT  # Y coords

print(f"   After: min={X_db_norm.min():.4f}, max={X_db_norm.max():.4f}")

# ========== LOAD Tey IMAGES FROM FOLDER ==========
print(f"\n🖼️  Loading Tey images from: {Tey_IMAGE_DIR}")

tey_features = []
img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
image_paths = []
for ext in img_extensions:
    image_paths.extend(glob.glob(os.path.join(Tey_IMAGE_DIR, ext)))

print(f"   Found {len(image_paths)} image files")

for i, img_path in enumerate(image_paths):
    feats = extract_landmarks_from_image(img_path)
    if feats is not None:
        tey_features.append(feats)
    if (i + 1) % 50 == 0:
        print(f"   Processed {i+1}/{len(image_paths)} images...")

print(f"   ✅ Extracted landmarks from {len(tey_features)} Tey images")

if len(tey_features) == 0:
    raise RuntimeError(
        "❌ No Tey images produced landmarks!\n"
        "   Make sure the Tey folder contains clear hand images.\n"
        f"   Folder: {Tey_IMAGE_DIR}"
    )

X_tey_images = np.array(tey_features, dtype=np.float32)

# ========== COMBINE Tey DATA (DB + images) ==========
tey_mask_db = y_db == TEY_LETTER
X_tey_db = X_db_norm[tey_mask_db]
y_tey_db = y_db[tey_mask_db]

print(f"\n📊 Tey samples from DB:    {len(X_tey_db)}")
print(f"   Tey samples from images: {len(X_tey_images)}")

if len(X_tey_db) > 0:
    X_tey = np.vstack([X_tey_db, X_tey_images])
    y_tey = np.array([TEY_LETTER] * len(X_tey))
else:
    X_tey = X_tey_images
    y_tey = np.array([TEY_LETTER] * len(X_tey))

print(f"   Total Tey samples: {len(X_tey)}")

# ========== AUGMENT Tey DATA ==========
print(f"\n🔧 Augmenting Tey data for robustness...")

augmented_X = [X_tey]
augmented_y = [y_tey]

# 1. Gaussian noise (camera noise)
for sigma in [0.01, 0.02]:
    noise = np.random.normal(0, sigma, X_tey.shape).astype(np.float32)
    augmented_X.append(np.clip(X_tey + noise, 0, 1))
    augmented_y.append(y_tey)

# 2. Scale (distance variation)
for scale in [0.9, 0.95, 1.05, 1.1]:
    augmented_X.append(np.clip(X_tey * scale, 0, 1))
    augmented_y.append(y_tey)

# 3. Translation (position variation)
for dx, dy in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02),
               (0.03, 0.03), (-0.03, -0.03)]:
    X_shift = X_tey.copy()
    X_shift[:, 0::2] = np.clip(X_shift[:, 0::2] + dx, 0, 1)
    X_shift[:, 1::2] = np.clip(X_shift[:, 1::2] + dy, 0, 1)
    augmented_X.append(X_shift)
    augmented_y.append(y_tey)

# 4. Small rotation (simulate hand tilt)
def rotate_landmarks(X, angle_deg):
    """Rotate (x,y) pairs around their centroid."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rot = X.copy()
    for i in range(0, X.shape[1], 2):
        x = X[:, i]
        y = X[:, i + 1]
        cx, cy = x.mean(), y.mean()
        xr = cx + (x - cx) * cos_a - (y - cy) * sin_a
        yr = cy + (x - cx) * sin_a + (y - cy) * cos_a
        X_rot[:, i] = np.clip(xr, 0, 1)
        X_rot[:, i + 1] = np.clip(yr, 0, 1)
    return X_rot

for ang in [-10, -5, 5, 10]:
    augmented_X.append(rotate_landmarks(X_tey, ang))
    augmented_y.append(y_tey)

X_tey_aug = np.vstack(augmented_X).astype(np.float32)
y_tey_aug = np.hstack(augmented_y)

print(f"   Augmented Tey samples: {len(X_tey_aug)}")

# ========== BUILD BINARY DATASET ==========
print(f"\n📊 Creating balanced dataset...")

X_other = X_db_norm[~tey_mask_db]
y_other = y_db[~tey_mask_db]

n_tey = len(X_tey_aug)
n_other = min(len(X_other), n_tey * 2)   # 2:1 negative-to-positive ratio

np.random.seed(42)
indices = np.random.choice(len(X_other), n_other, replace=False)
X_other_bal = X_other[indices]

X = np.vstack([X_tey_aug, X_other_bal])
y = np.hstack([np.ones(n_tey), np.zeros(n_other)]).astype(np.float32)

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} samples")
print(f"   Tey (1): {int(y.sum())} | Not Tey (0): {len(y) - int(y.sum())}")

# ========== SPLIT ==========
print(f"\n✂️ Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
print(f"\n🏗️ Building robust model...")

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
        os.path.join(OUTPUT_DIR, 'best_tey.h5'),
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

# Tey detection on ORIGINAL (non-augmented) samples
tey_preds = model.predict(X_tey, verbose=0).flatten()
tey_detected = (tey_preds > 0.5).sum()
tey_rate = tey_detected / len(X_tey) * 100
print(f"\n   Original Tey detection: {tey_rate:.1f}% ({tey_detected}/{len(X_tey)})")
print(f"   Avg Tey confidence: {tey_preds.mean():.4f}")
print(f"   Min Tey confidence: {tey_preds.min():.4f}")

# False positives on non-Tey data
other_preds = model.predict(X_other, verbose=0).flatten()
false_positives = (other_preds > 0.5).sum()
print(f"\n   False positives: {false_positives}/{len(X_other)} "
      f"({(1 - false_positives/len(X_other))*100:.1f}% specificity)")

# ========== SAVE ==========
print(f"\n💾 Saving models...")

model.save(os.path.join(OUTPUT_DIR, 'tey_robust.h5'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(OUTPUT_DIR, 'tey_robust.tflite'), 'wb') as f:
    f.write(tflite_model)

print(f"   TFLite size: {len(tflite_model)/1024:.1f} KB")

info = {
    'model': 'Tey Robust',
    'letter': TEY_LETTER,
    'sign_description': 'Closed fist with thumb extended upward (thumbs-up style)',
    'input': '42 MediaPipe landmarks (x,y in 0-1 range)',
    'preprocessing': 'NO scaler - raw MediaPipe values directly',
    'augmentation': 'Noise, Scale, Shift, Rotation',
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'tey_detection_rate': float(tey_rate),
    'image_size': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
    'threshold': 0.5,
    'classes': ['Not Tey', 'Tey']
}

with open(os.path.join(OUTPUT_DIR, 'tey_robust_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ TRAINING COMPLETE!")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Tey Detection: {tey_rate:.1f}%")
print(f"   Output dir:    {OUTPUT_DIR}")