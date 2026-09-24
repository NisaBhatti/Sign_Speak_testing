# Save as: model_training/train_bay_robust.py
# ACCURATE STRICT Bay (ب) detector - works with tilted hands

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
import cv2
import mediapipe as mp
from pathlib import Path
import glob

print("=" * 60)
print("SIGNSPEAK - Training STRICT Bay (ب) Model")
print("=" * 60)

# ========== YOUR PC PATHS ==========
DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
BAY_IMAGES_DIR = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Bay"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_bay"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
BAY_LETTER = 'ب'

mp_hands = mp.solutions.hands


# ========== FIXED GEOMETRIC RULE (TILT-TOLERANT) ==========
def is_strict_bay_pose(landmarks_42, verbose=False):
    """
    Check if landmarks match Bay sign:
    - 4 fingers extended (tip further from wrist than PIP, PIP further than MCP)
    - Fingers close together
    - Thumb bent (tip near index MCP)
    - Hand orientation roughly upward (not sideways / not fist)

    Uses DISTANCE FROM WRIST instead of absolute Y axis -> works for tilted hands.
    """
    lm = landmarks_42.reshape(21, 2)

    wrist = lm[0]
    thumb_tip = lm[4]; thumb_mcp = lm[2]
    index_mcp = lm[5];  index_pip = lm[6];  index_dip = lm[7];  index_tip = lm[8]
    middle_mcp = lm[9]; middle_pip = lm[10]; middle_dip = lm[11]; middle_tip = lm[12]
    ring_mcp = lm[13];  ring_pip = lm[14];  ring_dip = lm[15];  ring_tip = lm[16]
    pinky_mcp = lm[17]; pinky_pip = lm[18]; pinky_dip = lm[19]; pinky_tip = lm[20]

    # ---------- RULE 1: 4 fingers extended (distance from wrist increases) ----------
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def is_extended(mcp, pip, dip, tip):
        d_mcp = dist(mcp, wrist)
        d_pip = dist(pip, wrist)
        d_dip = dist(dip, wrist)
        d_tip = dist(tip, wrist)
        # Tip must be further from wrist than MCP by a healthy margin
        return (d_tip > d_mcp + 0.05) and (d_dip > d_pip - 0.02) and (d_pip > d_mcp - 0.02)

    index_up  = is_extended(index_mcp, index_pip, index_dip, index_tip)
    middle_up = is_extended(middle_mcp, middle_pip, middle_dip, middle_tip)
    ring_up   = is_extended(ring_mcp, ring_pip, ring_dip, ring_tip)
    pinky_up  = is_extended(pinky_mcp, pinky_pip, pinky_dip, pinky_tip)

    if not (index_up and middle_up and ring_up and pinky_up):
        if verbose:
            print(f"    Rule1 fail: idx={index_up} mid={middle_up} ring={ring_up} pink={pinky_up}")
        return False, "Not all 4 fingers extended"

    # ---------- RULE 2: Fingers roughly parallel (not spread wide) ----------
    tip_xs = [index_tip[0], middle_tip[0], ring_tip[0], pinky_tip[0]]
    spread = max(tip_xs) - min(tip_xs)
    if spread > 0.35:
        if verbose:
            print(f"    Rule2 fail: spread={spread:.3f}")
        return False, f"Fingers spread too wide ({spread:.2f})"

    # ---------- RULE 3: Thumb must be BENT (not sticking out sideways) ----------
    # Thumb tip should be CLOSE to index MCP
    thumb_to_index_mcp = dist(thumb_tip, index_mcp)
    if thumb_to_index_mcp > 0.22:
        if verbose:
            print(f"    Rule3 fail: thumb dist={thumb_to_index_mcp:.3f}")
        return False, f"Thumb not bent ({thumb_to_index_mcp:.2f})"

    # Thumb tip should NOT be further from wrist than thumb MCP (else it's extended)
    if dist(thumb_tip, wrist) > dist(thumb_mcp, wrist) + 0.10:
        if verbose:
            print(f"    Rule3b fail: thumb extended")
        return False, "Thumb extended, not bent"

    # ---------- RULE 4: Hand is upright (fingers point AWAY from wrist, generally up) ----------
    # Average direction from MCP to tip for 4 fingers should have negative Y (up on screen)
    dx_total = 0.0
    dy_total = 0.0
    for mcp, tip in [(index_mcp, index_tip), (middle_mcp, middle_tip),
                     (ring_mcp, ring_tip), (pinky_mcp, pinky_tip)]:
        dx_total += (tip[0] - mcp[0])
        dy_total += (tip[1] - mcp[1])

    avg_dx = dx_total / 4.0
    avg_dy = dy_total / 4.0

    # Fingers should point UP: dy must be negative (up in image coords)
    if avg_dy > -0.05:
        if verbose:
            print(f"    Rule4 fail: avg_dy={avg_dy:.3f} (fingers not pointing up)")
        return False, "Hand not upright"

    # Reject sideways hand: if horizontal displacement is greater than vertical
    if abs(avg_dx) > abs(avg_dy):
        if verbose:
            print(f"    Rule4b fail: too sideways dx={avg_dx:.3f} dy={avg_dy:.3f}")
        return False, "Hand too sideways"

    return True, "OK"


# ========== EXTRACT LANDMARKS ==========
def extract_landmarks_from_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    features = []
    for lm in results.multi_hand_landmarks[0].landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features, dtype=np.float32)


def extract_landmarks_from_folder(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"   ❌ Folder not found: {folder_path}")
        return np.array([])

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                  '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(str(folder / '**' / ext), recursive=True))
    image_paths = sorted(set(image_paths))
    print(f"   Found {len(image_paths)} images in {folder_path}")

    features_list = []
    success = 0
    failed = 0
    for i, img_path in enumerate(image_paths):
        features = extract_landmarks_from_image(img_path)
        if features is not None:
            features_list.append(features)
            success += 1
        else:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(image_paths)} | OK: {success} | No hand: {failed}")

    print(f"   ✅ Extracted: {success} | ❌ Failed: {failed}")
    if len(features_list) == 0:
        return np.array([])
    return np.array(features_list, dtype=np.float32)


# ========== LOAD BAY IMAGES ==========
print(f"\n📁 Loading Bay images from: {BAY_IMAGES_DIR}")
X_bay_raw = extract_landmarks_from_folder(BAY_IMAGES_DIR)

if len(X_bay_raw) == 0:
    print("\n❌ No Bay landmarks extracted. Check folder path & images.")
    exit(1)

print(f"   Total Bay samples: {len(X_bay_raw)}")

# ========== FILTER WITH FIXED RULE (with debug) ==========
print(f"\n🔍 Filtering Bay samples with tilt-tolerant geometric rule...")
valid_bay = []
rejected_reasons = {}

# Show verbose for first 3 images so you can debug
for i, feat in enumerate(X_bay_raw):
    verbose = (i < 3)
    if verbose:
        print(f"\n   -- Sample {i+1} --")
    ok, reason = is_strict_bay_pose(feat, verbose=verbose)
    if ok:
        valid_bay.append(feat)
    else:
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

print(f"\n   Rejection summary:")
for reason, count in sorted(rejected_reasons.items(), key=lambda x: -x[1]):
    print(f"     • {reason}: {count}")

valid_bay = np.array(valid_bay, dtype=np.float32) if len(valid_bay) > 0 else np.array([])
print(f"\n   ✅ Valid Bay poses: {len(valid_bay)} / {len(X_bay_raw)}")
print(f"   ❌ Rejected: {len(X_bay_raw) - len(valid_bay)}")

if len(valid_bay) == 0:
    print("\n❌ Still no valid poses. Now running in LENIENT mode (no rule filter)...")
    print("   This trains on all images but the live test still uses the rule.")
    valid_bay = X_bay_raw
    print(f"   Using all {len(valid_bay)} samples for training.")

X_bay_norm = valid_bay

# ========== LOAD NEGATIVES FROM DB ==========
print(f"\n📁 Loading negative samples from DB: {DB_PATH}")
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

X_db_norm = X_db.copy()
X_db_norm[:, 0::2] = X_db[:, 0::2] / IMAGE_WIDTH
X_db_norm[:, 1::2] = X_db[:, 1::2] / IMAGE_HEIGHT

non_bay_mask = y_db != BAY_LETTER
X_non_bay = X_db_norm[non_bay_mask]
print(f"   Non-Bay samples in DB: {len(X_non_bay)}")
print(f"   Classes: {sorted(set(y_db))}")

# ========== GENERATE SYNTHETIC NEGATIVES ==========
print(f"\n🔧 Generating synthetic negative poses (fists, open hands, wrong gestures)...")

def generate_synthetic_negatives(n_samples=4000, seed=42):
    np.random.seed(seed)
    samples = []

    for _ in range(n_samples):
        base = np.zeros(42, dtype=np.float32)
        wrist_x = 0.5 + np.random.uniform(-0.1, 0.1)
        wrist_y = 0.8
        base[0], base[1] = wrist_x, wrist_y

        pose_type = np.random.choice(['fist', 'open', 'point', 'curled', 'random'])

        for fi, x_center in enumerate([0.35, 0.45, 0.55, 0.65]):
            mcp_idx = 5 + fi * 4
            x = x_center + wrist_x - 0.5 + np.random.uniform(-0.05, 0.05)

            if pose_type == 'fist':
                # All curled
                ys = [0.65, 0.68, 0.72, 0.75]
            elif pose_type == 'open':
                # All extended, spread
                spread_offset = (fi - 1.5) * 0.08
                x += spread_offset
                ys = [0.60, 0.45, 0.35, 0.25]
            elif pose_type == 'point':
                # Only one finger extended
                if fi == 0:
                    ys = [0.60, 0.45, 0.35, 0.25]
                else:
                    ys = [0.65, 0.70, 0.74, 0.78]
            elif pose_type == 'curled':
                # Half curled
                ys = [0.65, 0.62, 0.65, 0.68]
            else:
                ys = [0.65] + list(np.random.uniform(0.35, 0.75, 3))

            for k, yy in enumerate(ys):
                base[(mcp_idx + k) * 2] = x + np.random.uniform(-0.02, 0.02)
                base[(mcp_idx + k) * 2 + 1] = yy + wrist_y - 0.8 + np.random.uniform(-0.02, 0.02)

        # Random thumb
        base[2*2] = wrist_x - 0.18 + np.random.uniform(-0.05, 0.05)
        base[2*2+1] = wrist_y - 0.05 + np.random.uniform(-0.05, 0.05)
        base[3*2] = wrist_x - 0.25 + np.random.uniform(-0.05, 0.05)
        base[3*2+1] = wrist_y - 0.10 + np.random.uniform(-0.05, 0.05)
        base[4*2] = wrist_x - 0.30 + np.random.uniform(-0.08, 0.08)
        base[4*2+1] = wrist_y - 0.15 + np.random.uniform(-0.08, 0.08)

        base += np.random.normal(0, 0.02, 42).astype(np.float32)
        base = np.clip(base, 0, 1)
        samples.append(base)

    return np.array(samples, dtype=np.float32)


X_synthetic = generate_synthetic_negatives(n_samples=4000)
print(f"   Generated synthetic negatives: {len(X_synthetic)}")

# ========== AUGMENT BAY DATA ==========
print(f"\n🔧 Augmenting Bay data...")
augmented_X = [X_bay_norm]
augmented_y = [np.ones(len(X_bay_norm))]

for _ in range(4):
    noise = np.random.normal(0, 0.015, X_bay_norm.shape).astype(np.float32)
    augmented_X.append(np.clip(X_bay_norm + noise, 0, 1))
    augmented_y.append(np.ones(len(X_bay_norm)))

for scale in [0.88, 0.92, 0.96, 1.04, 1.08, 1.12]:
    augmented_X.append(np.clip(X_bay_norm * scale, 0, 1))
    augmented_y.append(np.ones(len(X_bay_norm)))

for sx, sy in [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02),
                (0.03, 0.03), (-0.03, -0.03), (0.03, -0.03), (-0.03, 0.03)]:
    Xs = X_bay_norm.copy()
    Xs[:, 0::2] = np.clip(Xs[:, 0::2] + sx, 0, 1)
    Xs[:, 1::2] = np.clip(Xs[:, 1::2] + sy, 0, 1)
    augmented_X.append(Xs)
    augmented_y.append(np.ones(len(X_bay_norm)))

X_flipped = X_bay_norm.copy()
X_flipped[:, 0::2] = 1.0 - X_flipped[:, 0::2]
augmented_X.append(X_flipped)
augmented_y.append(np.ones(len(X_bay_norm)))

X_bay_augmented = np.vstack(augmented_X)
y_bay_augmented = np.hstack(augmented_y)
print(f"   Augmented Bay samples: {len(X_bay_augmented)}")

# ========== BUILD DATASET ==========
print(f"\n📊 Creating final dataset...")
n_bay = len(X_bay_augmented)

X_negatives_all = np.vstack([X_non_bay, X_synthetic])
y_negatives_all = np.zeros(len(X_negatives_all))

n_neg = min(len(X_negatives_all), n_bay * 2)
np.random.seed(42)
idx = np.random.choice(len(X_negatives_all), n_neg, replace=False)
X_negatives_bal = X_negatives_all[idx]
y_negatives_bal = y_negatives_all[idx]

X = np.vstack([X_bay_augmented, X_negatives_bal])
y = np.hstack([y_bay_augmented, y_negatives_bal])

shuffle = np.random.permutation(len(X))
X, y = X[shuffle], y[shuffle]

print(f"   Total: {len(X)} samples")
print(f"   Bay (1): {int(y.sum())} | Not-Bay (0): {len(y) - int(y.sum())}")

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
class_weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}
print(f"\n⚖️ Class weights: {class_weight_dict}")

# ========== MODEL ==========
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
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_bay.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=32,
    callbacks=callbacks, class_weight=class_weight_dict, verbose=1
)

# ========== EVALUATE ==========
print(f"\n📊 Results:")
train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"   Train: Acc={train_acc*100:.2f}%, AUC={train_auc:.4f}")
print(f"   Val:   Acc={val_acc*100:.2f}%, AUC={val_auc:.4f}")
print(f"   Test:  Acc={test_acc*100:.2f}%, AUC={test_auc:.4f}")

bay_preds = model.predict(X_bay_norm, verbose=0).flatten()
bay_detected = (bay_preds > 0.5).sum()
bay_rate = bay_detected / len(X_bay_norm) * 100
print(f"\n   Bay detection (original): {bay_rate:.1f}% ({bay_detected}/{len(X_bay_norm)})")

non_bay_preds = model.predict(X_negatives_all, verbose=0).flatten()
fp = (non_bay_preds > 0.5).sum()
spec = (1 - fp / len(X_negatives_all)) * 100
print(f"   Specificity (reject non-Bay): {spec:.1f}% ({fp} false positives)")

# ========== SAVE ==========
print(f"\n💾 Saving to: {OUTPUT_DIR}")
model.save(os.path.join(OUTPUT_DIR, 'bay_robust.h5'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(OUTPUT_DIR, 'bay_robust.tflite'), 'wb') as f:
    f.write(tflite_model)
print(f"   TFLite size: {len(tflite_model)/1024:.1f} KB")

info = {
    'model': 'Bay STRICT v2',
    'letter': BAY_LETTER,
    'sign_description': '4 fingers straight up, thumb bent across palm',
    'geometric_rule': 'YES (tilt-tolerant)',
    'recommended_threshold': 0.85,
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'bay_detection_rate': float(bay_rate),
    'specificity': float(spec),
    'classes': ['Not Bay', 'Bay']
}
with open(os.path.join(OUTPUT_DIR, 'bay_robust_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ TRAINING COMPLETE!")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Bay Detection: {bay_rate:.1f}%")
print(f"   Specificity:   {spec:.1f}%")