# Save as: model_training/train_pay_robust.py

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

print("="*60)
print("SIGNSPEAK - Training Pay (پ) VERTICAL ONLY Model")
print("="*60)

# ========== CONFIGURATION ==========
DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_pay"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
PAY_LETTER = 'پ'

# ========== NORMALIZATION (NO ROTATION - KEEPS ORIENTATION) ==========
def normalize_landmarks_keep_orientation(features):
    """
    Only centers and scales the hand. DOES NOT rotate.
    This keeps the hand's original orientation so vertical vs horizontal matters.
    """
    pts = np.array(features).reshape(21, 2)
    
    # 1. Translation: Center at wrist (landmark 0)
    wrist = pts[0]
    pts = pts - wrist
    
    # 2. Scale: Divide by max distance from wrist
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist
    
    # NO rotation step - orientation is preserved
    return pts.flatten()

# ========== LOAD DATA ==========
print("\n📁 Loading all data...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM rightHandDataset")
all_data = cursor.fetchall()
conn.close()
print(f"   Total samples: {len(all_data)}")

X_all = []
y_all = []

for row in all_data:
    raw_features = [float(row[i]) for i in range(1, 43)]
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    features = normalize_landmarks_keep_orientation(raw_features)
    X_all.append(features)
    y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all)

# ========== RESCALE TO 0-1 ==========
print(f"\n📏 Rescaling features...")
X_norm = (X_all + 1.0) / 2.0
X_norm = np.clip(X_norm, 0, 1)
print(f"   Range: min={X_norm.min():.4f}, max={X_norm.max():.4f}")

# ========== AUGMENTATION (NO ROTATION AUGMENTATION!) ==========
print(f"\n🔧 Augmenting Pay data (orientation preserved)...")

pay_mask = y_all == PAY_LETTER
X_pay_original = X_norm[pay_mask].copy()
y_pay_original = y_all[pay_mask].copy()

print(f"   Original Pay samples: {len(X_pay_original)}")

if len(X_pay_original) == 0:
    print("\n❌ ERROR: No Pay (پ) samples found!")
    exit()

augmented_X = [X_pay_original]
augmented_y = [y_pay_original]

# Only light noise, scale, and small shifts - NO rotation
noise = np.random.normal(0, 0.02, X_pay_original.shape).astype(np.float32)
X_noisy = np.clip(X_pay_original + noise, 0, 1)
augmented_X.append(X_noisy)
augmented_y.append(y_pay_original)

for scale in [0.9, 1.1]:
    X_scaled = np.clip(X_pay_original * scale, 0, 1)
    augmented_X.append(X_scaled)
    augmented_y.append(y_pay_original)

# Small shifts only (keeps vertical orientation)
for shift_x, shift_y in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
    X_shifted = X_pay_original.copy()
    X_shifted[:, 0::2] = np.clip(X_shifted[:, 0::2] + shift_x, 0, 1)
    X_shifted[:, 1::2] = np.clip(X_shifted[:, 1::2] + shift_y, 0, 1)
    augmented_X.append(X_shifted)
    augmented_y.append(y_pay_original)

X_pay_augmented = np.vstack(augmented_X)
y_pay_augmented = np.hstack(augmented_y)
print(f"   Augmented Pay samples: {len(X_pay_augmented)}")

# ========== CREATE DATASET ==========
print(f"\n📊 Creating balanced dataset...")

X_other = X_norm[~pay_mask]
y_other = y_all[~pay_mask]

n_pay = len(X_pay_augmented)
n_other = min(len(X_other), n_pay * 2)

np.random.seed(42)
indices = np.random.choice(len(X_other), n_other, replace=False)
X_other_bal = X_other[indices]
y_other_bal = y_other[indices]

X = np.vstack([X_pay_augmented, X_other_bal])
y = np.hstack([np.ones(n_pay), np.zeros(n_other)])

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} | Pay: {int(y.sum())} | Other: {len(y)-int(y.sum())}")

# ========== SPLIT ==========
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

# ========== BUILD MODEL ==========
print(f"\n🏗️ Building model...")

model = Sequential([
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(), Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(), Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(16, activation='relu'),
    BatchNormalization(), Dropout(0.2),
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# ========== TRAIN ==========
print(f"\n🚀 Training...")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_pay.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=32,
    callbacks=callbacks, class_weight=class_weight_dict, verbose=1
)

# ========== EVALUATE ==========
print(f"\n📊 Final Results:")
train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"   Train: Acc={train_acc*100:.2f}%, AUC={train_auc:.4f}")
print(f"   Val:   Acc={val_acc*100:.2f}%, AUC={val_auc:.4f}")
print(f"   Test:  Acc={test_acc*100:.2f}%, AUC={test_auc:.4f}")

pay_preds = model.predict(X_pay_original, verbose=0).flatten()
pay_rate = (pay_preds > 0.5).sum() / len(X_pay_original) * 100
print(f"\n   Original Pay detection: {pay_rate:.1f}%")

other_preds = model.predict(X_other, verbose=0).flatten()
false_positives = (other_preds > 0.5).sum()
print(f"   False positives: {false_positives}/{len(X_other)}")

# ========== SAVE ==========
print(f"\n💾 Saving models...")
model.save(os.path.join(OUTPUT_DIR, 'pay_robust.h5'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(OUTPUT_DIR, 'pay_robust.tflite'), 'wb') as f:
    f.write(tflite_model)

info = {
    'model': 'Pay Vertical-Only',
    'letter': PAY_LETTER,
    'input': '42 MediaPipe landmarks (translation + scale normalized only)',
    'preprocessing': 'NO rotation - orientation preserved',
    'note': 'Only detects Pay when hand is VERTICAL like training image',
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'pay_detection_rate': float(pay_rate),
    'threshold': 0.5,
    'classes': ['Not Pay', 'Pay']
}

with open(os.path.join(OUTPUT_DIR, 'pay_robust_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ TRAINING COMPLETE!")
print(f"   Test Accuracy: {test_acc*100:.2f}%")