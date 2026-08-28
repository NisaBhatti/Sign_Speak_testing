# Save as: model_training/train_alif_robust.py

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
print("SIGNSPEAK - Training Alif ROBUST Model")
print("="*60)

# ========== CONFIGURATION ==========
DB_PATH = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\dataset\main_dataset.db"
OUTPUT_DIR = r"C:\Users\asifa\OneDrive\Desktop\Final Year Project\model_training\exported_models_alif"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

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
    features = [float(row[i]) for i in range(1, 43)]
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', row[43]).strip()
    X_all.append(features)
    y_all.append(label)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all)

# ========== NORMALIZE ==========
print(f"\n📏 Normalizing features...")
print(f"   Before: min={X_all.min():.1f}, max={X_all.max():.1f}")

X_norm = X_all.copy()
X_norm[:, 0::2] = X_all[:, 0::2] / IMAGE_WIDTH   # X coordinates
X_norm[:, 1::2] = X_all[:, 1::2] / IMAGE_HEIGHT  # Y coordinates

print(f"   After: min={X_norm.min():.4f}, max={X_norm.max():.4f}")
print(f"   Mean: {X_norm.mean():.4f}, Std: {X_norm.std():.4f}")

# ========== DATA AUGMENTATION FOR ALIF ==========
print(f"\n🔧 Augmenting Alif data for robustness...")

alif_mask = y_all == 'ا'
X_alif_original = X_norm[alif_mask].copy()
y_alif_original = y_all[alif_mask].copy()

print(f"   Original Alif samples: {len(X_alif_original)}")

# Create augmented versions
augmented_X = [X_alif_original]
augmented_y = [y_alif_original]

# Augmentation 1: Add small random noise (simulates camera noise)
noise = np.random.normal(0, 0.02, X_alif_original.shape).astype(np.float32)
X_noisy = np.clip(X_alif_original + noise, 0, 1)
augmented_X.append(X_noisy)
augmented_y.append(y_alif_original)

# Augmentation 2: Scale slightly (simulates different distances)
for scale in [0.9, 1.1]:
    X_scaled = np.clip(X_alif_original * scale, 0, 1)
    augmented_X.append(X_scaled)
    augmented_y.append(y_alif_original)

# Augmentation 3: Shift slightly (simulates different positions)
for shift_x, shift_y in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
    X_shifted = X_alif_original.copy()
    X_shifted[:, 0::2] = np.clip(X_shifted[:, 0::2] + shift_x, 0, 1)  # Shift X
    X_shifted[:, 1::2] = np.clip(X_shifted[:, 1::2] + shift_y, 0, 1)  # Shift Y
    augmented_X.append(X_shifted)
    augmented_y.append(y_alif_original)

# Combine all augmentations
X_alif_augmented = np.vstack(augmented_X)
y_alif_augmented = np.hstack(augmented_y)

print(f"   Augmented Alif samples: {len(X_alif_augmented)}")

# ========== CREATE DATASET ==========
print(f"\n📊 Creating balanced dataset...")

# Get other class samples
X_other = X_norm[~alif_mask]
y_other = y_all[~alif_mask]

# Take enough other samples to balance
n_alif = len(X_alif_augmented)
n_other = min(len(X_other), n_alif * 2)

np.random.seed(42)
indices = np.random.choice(len(X_other), n_other, replace=False)
X_other_bal = X_other[indices]
y_other_bal = y_other[indices]

# Combine
X = np.vstack([X_alif_augmented, X_other_bal])
y = np.hstack([np.ones(n_alif), np.zeros(n_other)])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} samples")
print(f"   Alif (1): {int(y.sum())} | Other (0): {len(y)-int(y.sum())}")

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
    # Input layer
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.5),
    
    # Hidden layers
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
    
    # Output
    Dense(1, activation='sigmoid')
])

# Use lower learning rate for better convergence
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
        os.path.join(OUTPUT_DIR, 'best_alif.h5'),
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

# Test on original Alif samples (not augmented)
alif_preds = model.predict(X_alif_original, verbose=0).flatten()
alif_detected = (alif_preds > 0.5).sum()
alif_rate = alif_detected / len(X_alif_original) * 100
print(f"\n   Original Alif detection: {alif_rate:.1f}% ({alif_detected}/{len(X_alif_original)})")
print(f"   Avg Alif confidence: {alif_preds.mean():.4f}")
print(f"   Min Alif confidence: {alif_preds.min():.4f}")
print(f"   Max Alif confidence: {alif_preds.max():.4f}")

# Test on other samples
other_preds = model.predict(X_other, verbose=0).flatten()
false_positives = (other_preds > 0.5).sum()
print(f"\n   False positives: {false_positives}/{len(X_other)} ({(1-false_positives/len(X_other))*100:.1f}% specificity)")

# ========== SAVE ==========
print(f"\n💾 Saving models...")

# Keras model
model.save(os.path.join(OUTPUT_DIR, 'alif_robust.h5'))

# TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(OUTPUT_DIR, 'alif_robust.tflite'), 'wb') as f:
    f.write(tflite_model)

print(f"   TFLite size: {len(tflite_model)/1024:.1f} KB")

# Info
info = {
    'model': 'Alif Robust',
    'input': '42 MediaPipe landmarks (x,y in 0-1 range)',
    'preprocessing': 'NO scaler - raw MediaPipe values directly',
    'augmentation': 'Noise, Scale, Shift applied',
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'alif_detection_rate': float(alif_rate),
    'image_size': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
    'threshold': 0.5,
    'classes': ['Not Alif', 'Alif']
}

with open(os.path.join(OUTPUT_DIR, 'alif_robust_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ TRAINING COMPLETE!")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Alif Detection: {alif_rate:.1f}%")