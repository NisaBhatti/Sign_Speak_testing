# Save as: model_training/train_lam_robust.py

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
print("SIGNSPEAK - Training Lām (ل) ROBUST Model")
print("="*60)

# ========== CONFIGURATION ==========
DB_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\main_dataset.db"
OUTPUT_DIR = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model\exported_models_lam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

# Arabic/Urdu letter Lām — U+0644
LAM_LETTER = 'ل'

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

# ========== FILTER FOR LAM ==========
print(f"\n🔎 Filtering samples for '{LAM_LETTER}' (U+0644)...")

lam_mask = y_all == LAM_LETTER
X_lam_original = X_norm[lam_mask].copy()
y_lam_original = y_all[lam_mask].copy()

print(f"   Original Lām samples: {len(X_lam_original)}")

# ========== SAFETY CHECK ==========
if len(X_lam_original) == 0:
    print(f"\n❌ ERROR: No samples found for letter '{LAM_LETTER}' (U+0644)!")
    print(f"   Available labels in DB:")
    for lbl in sorted(set(y_all)):
        codepoints = " ".join(f"U+{ord(c):04X}" for c in lbl)
        print(f"      {repr(lbl)}  ->  {codepoints}")
    print(f"\n   👉 Change LAM_LETTER to match one of the labels above.")
    print(f"   👉 Common variants: ل (U+0644), ا (U+0627), لا (U+0644+U+0627)")
    exit(1)

if len(X_lam_original) < 10:
    print(f"\n⚠️  WARNING: Only {len(X_lam_original)} samples found.")
    print(f"   Training may be unreliable. Consider collecting more data.")

# ========== DATA AUGMENTATION ==========
print(f"\n🔧 Augmenting Lām data for robustness...")

augmented_X = [X_lam_original]
augmented_y = [y_lam_original]

# Augmentation 1: Add small random noise (simulates camera noise)
noise = np.random.normal(0, 0.02, X_lam_original.shape).astype(np.float32)
X_noisy = np.clip(X_lam_original + noise, 0, 1)
augmented_X.append(X_noisy)
augmented_y.append(y_lam_original)

# Augmentation 2: Scale slightly (simulates different distances)
for scale in [0.9, 1.1]:
    X_scaled = np.clip(X_lam_original * scale, 0, 1)
    augmented_X.append(X_scaled)
    augmented_y.append(y_lam_original)

# Augmentation 3: Shift slightly (simulates different positions)
for shift_x, shift_y in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]:
    X_shifted = X_lam_original.copy()
    X_shifted[:, 0::2] = np.clip(X_shifted[:, 0::2] + shift_x, 0, 1)
    X_shifted[:, 1::2] = np.clip(X_shifted[:, 1::2] + shift_y, 0, 1)
    augmented_X.append(X_shifted)
    augmented_y.append(y_lam_original)

# Combine all augmentations
X_lam_augmented = np.vstack(augmented_X)
y_lam_augmented = np.hstack(augmented_y)

print(f"   Augmented Lām samples: {len(X_lam_augmented)}")

# ========== CREATE BALANCED DATASET ==========
print(f"\n📊 Creating balanced dataset...")

X_other = X_norm[~lam_mask]
y_other = y_all[~lam_mask]

n_lam = len(X_lam_augmented)
n_other = min(len(X_other), n_lam * 2)

np.random.seed(42)
indices = np.random.choice(len(X_other), n_other, replace=False)
X_other_bal = X_other[indices]
y_other_bal = y_other[indices]

X = np.vstack([X_lam_augmented, X_other_bal])
y = np.hstack([np.ones(n_lam), np.zeros(n_other)])

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} samples")
print(f"   Lām (1): {int(y.sum())} | Other (0): {len(y)-int(y.sum())}")

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
        os.path.join(OUTPUT_DIR, 'best_lam.h5'),
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

# Test on ORIGINAL Lām samples
lam_preds = model.predict(X_lam_original, verbose=0).flatten()
lam_detected = (lam_preds > 0.5).sum()
lam_rate = lam_detected / len(X_lam_original) * 100

print(f"\n   Original Lām detection: {lam_rate:.1f}% ({lam_detected}/{len(X_lam_original)})")
print(f"   Avg Lām confidence: {lam_preds.mean():.4f}")
print(f"   Min Lām confidence: {lam_preds.min():.4f}")
print(f"   Max Lām confidence: {lam_preds.max():.4f}")

# Test on other samples (specificity)
other_preds = model.predict(X_other, verbose=0).flatten()
false_positives = (other_preds > 0.5).sum()
specificity = (1 - false_positives / len(X_other)) * 100

print(f"\n   False positives: {false_positives}/{len(X_other)} ({specificity:.1f}% specificity)")

# ========== SAVE MODELS ==========
print(f"\n💾 Saving models...")

keras_path = os.path.join(OUTPUT_DIR, 'lam_robust.h5')
model.save(keras_path)
print(f"   Saved: {keras_path}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, 'lam_robust.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"   Saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

info = {
    'model': 'Lam Robust',
    'letter': LAM_LETTER,
    'unicode': 'U+0644',
    'language': 'Arabic / Urdu / Persian (Lām)',
    'input': '42 MediaPipe landmarks (x,y in 0-1 range)',
    'preprocessing': 'NO scaler - raw MediaPipe values directly',
    'augmentation': 'Noise, Scale, Shift applied',
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'lam_detection_rate': float(lam_rate),
    'specificity': float(specificity / 100),
    'image_size': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}',
    'threshold': 0.5,
    'classes': ['Not Lam', 'Lam'],
    'n_original_samples': int(len(X_lam_original)),
    'n_augmented_samples': int(len(X_lam_augmented))
}

info_path = os.path.join(OUTPUT_DIR, 'lam_robust_info.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"   Saved: {info_path}")

# ========== DONE ==========
print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"   Letter:           {LAM_LETTER}  (U+0644) — Lām")
print(f"   Test Accuracy:    {test_acc*100:.2f}%")
print(f"   Test AUC:         {test_auc:.4f}")
print(f"   Lām Detection:    {lam_rate:.1f}%")
print(f"   Specificity:      {specificity:.1f}%")
print(f"   Output folder:    {OUTPUT_DIR}")
print(f"{'='*60}")
print(f"\n👉 Next: run testing/test_lam_robust.py")