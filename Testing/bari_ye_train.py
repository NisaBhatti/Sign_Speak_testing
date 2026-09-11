# Save as: model_training/train_bari_ye_robust.py

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
print("SIGNSPEAK - Training Baṛī Ye (ے) ROBUST Model")
print("="*60)

# ========== CONFIGURATION ==========
DB_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\main_dataset.db"
OUTPUT_DIR = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_bari_ye"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

# Urdu letter Baṛī Ye (Yeh Barree) — U+06D2
BARI_YE_LETTER = 'ے'

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

# ========== FILTER FOR BARI YE ==========
print(f"\n🔎 Filtering samples for '{BARI_YE_LETTER}' (U+06D2)...")

bari_ye_mask = y_all == BARI_YE_LETTER
X_bari_ye_original = X_norm[bari_ye_mask].copy()
y_bari_ye_original = y_all[bari_ye_mask].copy()

print(f"   Original Baṛī Ye samples: {len(X_bari_ye_original)}")

# ========== SAFETY CHECK ==========
if len(X_bari_ye_original) == 0:
    print(f"\n❌ ERROR: No samples found for letter '{BARI_YE_LETTER}' (U+06D2)!")
    print(f"   Available labels in DB:")
    for lbl in sorted(set(y_all)):
        codepoints = " ".join(f"U+{ord(c):04X}" for c in lbl)
        print(f"      {repr(lbl)}  ->  {codepoints}")
    print(f"\n   👉 Change BARI_YE_LETTER to match one of the labels above.")
    exit(1)

if len(X_bari_ye_original) < 10:
    print(f"\n⚠️  WARNING: Only {len(X_bari_ye_original)} samples found.")
    print(f"   Training may be unreliable. Consider collecting more data.")

# ========== DATA AUGMENTATION ==========
print(f"\n🔧 Augmenting Baṛī Ye data for robustness...")

augmented_X = [X_bari_ye_original]
augmented_y = [y_bari_ye_original]

# Augmentation 1: Add small random noise (simulates camera noise)
noise = np.random.normal(0, 0.02, X_bari_ye_original.shape).astype(np.float32)
X_noisy = np.clip(X_bari_ye_original + noise, 0, 1)
augmented_X.append(X_noisy)
augmented_y.append(y_bari_ye_original)

# Augmentation 2: Scale slightly (simulates different distances)
for scale in [0.9, 1.1]:
    X_scaled = np.clip(X_bari_ye_original * scale, 0, 1)
    augmented_X.append(X_scaled)
    augmented_y.append(y_bari_ye_original)

# Augmentation 3: Shift slightly (simulates different positions)
for shift_x, shift_y in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
    X_shifted = X_bari_ye_original.copy()
    X_shifted[:, 0::2] = np.clip(X_shifted[:, 0::2] + shift_x, 0, 1)  # Shift X
    X_shifted[:, 1::2] = np.clip(X_shifted[:, 1::2] + shift_y, 0, 1)  # Shift Y
    augmented_X.append(X_shifted)
    augmented_y.append(y_bari_ye_original)

# Combine all augmentations
X_bari_ye_augmented = np.vstack(augmented_X)
y_bari_ye_augmented = np.hstack(augmented_y)

print(f"   Augmented Baṛī Ye samples: {len(X_bari_ye_augmented)}")

# ========== CREATE BALANCED DATASET ==========
print(f"\n📊 Creating balanced dataset...")

X_other = X_norm[~bari_ye_mask]
y_other = y_all[~bari_ye_mask]

# Balance: use 2x other samples for each Baṛī Ye sample
n_bari_ye = len(X_bari_ye_augmented)
n_other = min(len(X_other), n_bari_ye * 2)

np.random.seed(42)
indices = np.random.choice(len(X_other), n_other, replace=False)
X_other_bal = X_other[indices]
y_other_bal = y_other[indices]

# Combine
X = np.vstack([X_bari_ye_augmented, X_other_bal])
y = np.hstack([np.ones(n_bari_ye), np.zeros(n_other)])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} samples")
print(f"   Baṛī Ye (1): {int(y.sum())} | Other (0): {len(y)-int(y.sum())}")

# ========== TRAIN / VAL / TEST SPLIT ==========
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
class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
print(f"\n⚖️ Class weights: {class_weight_dict}")

# ========== BUILD MODEL ==========
print(f"\n🏗️ Building robust model...")

model = Sequential([
    # Input layer
    Dense(256, activation='relu', input_shape=(42,)),
    BatchNormalization(),
    Dropout(0.5),

    # Hidden layer 1
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    # Hidden layer 2
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden layer 3
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden layer 4
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden layer 5
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden layer 6
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Output (binary)
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
    EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_bari_ye.h5'),
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

# Test on ORIGINAL Baṛī Ye samples (not augmented)
bari_ye_preds = model.predict(X_bari_ye_original, verbose=0).flatten()
bari_ye_detected = (bari_ye_preds > 0.5).sum()
bari_ye_rate = bari_ye_detected / len(X_bari_ye_original) * 100

print(f"\n   Original Baṛī Ye detection: {bari_ye_rate:.1f}% ({bari_ye_detected}/{len(X_bari_ye_original)})")
print(f"   Avg Baṛī Ye confidence: {bari_ye_preds.mean():.4f}")
print(f"   Min Baṛī Ye confidence: {bari_ye_preds.min():.4f}")
print(f"   Max Baṛī Ye confidence: {bari_ye_preds.max():.4f}")

# Test on other samples (specificity)
other_preds = model.predict(X_other, verbose=0).flatten()
false_positives = (other_preds > 0.5).sum()
specificity = (1 - false_positives / len(X_other)) * 100

print(f"\n   False positives: {false_positives}/{len(X_other)} ({specificity:.1f}% specificity)")

# ========== SAVE MODELS ==========
print(f"\n💾 Saving models...")

# 1) Full Keras model
keras_path = os.path.join(OUTPUT_DIR, 'bari_ye_robust.h5')
model.save(keras_path)
print(f"   Saved: {keras_path}")

# 2) TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, 'bari_ye_robust.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"   Saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

# 3) Info JSON
info = {
    'model': 'Bari Ye Robust',
    'letter': BARI_YE_LETTER,
    'unicode': 'U+06D2',
    'language': 'Urdu / Punjabi / Sindhi (Yeh Barree)',
    'input': '42 MediaPipe landmarks (x,y in 0-1 range)',
    'preprocessing': 'NO scaler - raw MediaPipe values directly',
    'augmentation': 'Noise, Scale, Shift applied',
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'bari_ye_detection_rate': float(bari_ye_rate),
    'specificity': float(specificity / 100),
    'image_size': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
    'threshold': 0.5,
    'classes': ['Not Bari Ye', 'Bari Ye'],
    'n_original_samples': int(len(X_bari_ye_original)),
    'n_augmented_samples': int(len(X_bari_ye_augmented))
}

info_path = os.path.join(OUTPUT_DIR, 'bari_ye_robust_info.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"   Saved: {info_path}")

# ========== DONE ==========
print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"   Letter:           {BARI_YE_LETTER}  (U+06D2) — Baṛī Ye")
print(f"   Test Accuracy:    {test_acc*100:.2f}%")
print(f"   Test AUC:         {test_auc:.4f}")
print(f"   Baṛī Ye Detection:{bari_ye_rate:.1f}%")
print(f"   Specificity:      {specificity:.1f}%")
print(f"   Output folder:    {OUTPUT_DIR}")
print(f"{'='*60}")
print(f"\n👉 Next: run the testing script:")
print(f"   testing/test_bari_ye_robust.py")