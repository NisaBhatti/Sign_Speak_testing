# Save as: train_tay.py
# Robust binary Tay (ت) detector
# Key fixes: wrist-relative normalization, scale invariance,
#            handedness alignment, verified Tay landmark extraction.

import sqlite3, os, re, glob, json, sys
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ============ CONFIG ============
TAY_IMAGE_DIR = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Tay"
OTHER_DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
OUTPUT_DIR    = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_tay"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_W, IMG_H = 300, 300
TAY_LETTER   = 'ت'
THRESHOLD    = 0.5

clean = lambda s: re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', str(s)).strip()

print("="*60)
print("SIGNSPEAK - Tay (ت) Training (robust)")
print("="*60)

# ============================================================
# CORE: Wrist-relative, scale-invariant, handedness-aware norm
# Returns 42 floats, all in a consistent coordinate frame.
# ============================================================
WRIST = 0
MID_MCP = 9   # middle finger MCP — good stable scale reference

def normalize_landmarks(landmarks_xy, handedness):
    """
    landmarks_xy: list of 21 (x,y) tuples in image pixel coords (or 0-1).
    handedness:   'Left' or 'Right' (from MediaPipe, image POV).
    Returns flat 42-vector: wrist-relative, scale-normalized, mirrored
    so that the hand is ALWAYS in the same canonical orientation.
    """
    pts = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)

    # 1) Translate so wrist is at origin
    pts = pts - pts[WRIST]

    # 2) Mirror if right hand, so all hands look like a "left hand"
    #    (MediaPipe labels from image POV; use consistent rule)
    if handedness == 'Right':
        pts[:, 0] = -pts[:, 0]

    # 3) Scale by wrist->middle_mcp distance (robust to hand size)
    scale = np.linalg.norm(pts[MID_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale

    return pts.flatten().astype(np.float32)

def extract_from_image(path):
    """Returns (features 42, handedness) or (None, None)."""
    img = cv2.imread(path)
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None
    lm = res.multi_hand_landmarks[0]
    handedness = res.multi_handedness[0].classification[0].label  # 'Left'/'Right'
    pts = [(p.x, p.y) for p in lm.landmark]
    return normalize_landmarks(pts, handedness), handedness

# ============ STEP 1: EXTRACT TAY FROM IMAGES ============
print(f"\n📁 Extracting Tay landmarks from: {TAY_IMAGE_DIR}")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.3, min_tracking_confidence=0.3)

paths = []
for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG'):
    paths.extend(glob.glob(os.path.join(TAY_IMAGE_DIR, ext)))
paths = sorted(set(paths))
print(f"   Found {len(paths)} image files")

X_tay, hands_seen = [], {'Left': 0, 'Right': 0}
for i, p in enumerate(paths, 1):
    feats, hd = extract_from_image(p)
    if feats is None:
        print(f"   [{i}/{len(paths)}] ❌ No hand: {os.path.basename(p)}")
        continue
    X_tay.append(feats)
    hands_seen[hd] = hands_seen.get(hd, 0) + 1

hands.close()
X_tay = np.array(X_tay, dtype=np.float32)
print(f"   ✅ Valid Tay samples: {len(X_tay)}")
print(f"   Handedness in Tay folder: {hands_seen}")
if len(X_tay) < 20:
    sys.exit("❌ Need at least 20 valid Tay images. Check folder / try again.")

# ============ STEP 2: LOAD NON-TAY FROM main_dataset.db ============
print(f"\n📁 Loading non-Tay samples from main DB...")
conn = sqlite3.connect(OTHER_DB_PATH)
rows = conn.cursor().execute("SELECT * FROM rightHandDataset").fetchall()
conn.close()

X_other_raw = []
for row in rows:
    if clean(row[43]) == TAY_LETTER:
        continue
    X_other_raw.append([float(row[i]) for i in range(1, 43)])
X_other_raw = np.array(X_other_raw, dtype=np.float32)
print(f"   Non-Tay samples available: {len(X_other_raw)}")

# ============================================================
# IMPORTANT: main_dataset.db stores raw MediaPipe coords normalized
# to (0-1). We must convert them through the SAME pipeline.
# But we don't have handedness stored, and DB values are x,y already
# in 0-1 form. Apply translation+scale to match Tay convention.
# ============================================================
def normalize_db_row(flat42):
    """Apply wrist-relative + scale normalization to a DB row."""
    pts = np.asarray(flat42, dtype=np.float32).reshape(21, 2)
    pts = pts - pts[WRIST]
    scale = np.linalg.norm(pts[MID_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale
    return pts.flatten().astype(np.float32)

X_other = np.stack([normalize_db_row(r) for r in X_other_raw])

# ============ STEP 3: AUGMENT TAY (safe transforms only) ============
print("\n🔧 Augmenting Tay samples...")
rng = np.random.default_rng(42)
aug_X = [X_tay]
aug_y = [np.ones(len(X_tay), dtype=np.float32)]

# Small Gaussian noise
for sigma in (0.02, 0.04):
    aug_X.append(X_tay + rng.normal(0, sigma, X_tay.shape).astype(np.float32))
    aug_y.append(np.ones(len(X_tay), dtype=np.float32))

# Small uniform scale (0.9–1.1) — already scale-invariant, but jitter helps
for s in (0.9, 0.95, 1.05, 1.1):
    aug_X.append(X_tay * s); aug_y.append(np.ones(len(X_tay), dtype=np.float32))

# Small 2D rotation around origin (wrist)
for ang in (-10, -5, 5, 10):
    t = np.deg2rad(ang); c, s = np.cos(t), np.sin(t)
    pts = X_tay.reshape(-1, 21, 2)
    r = np.stack([pts[...,0]*c - pts[...,1]*s, pts[...,0]*s + pts[...,1]*c], axis=-1)
    aug_X.append(r.reshape(-1, 42).astype(np.float32))
    aug_y.append(np.ones(len(X_tay), dtype=np.float32))

# Per-landmark jitter (small)
for _ in range(3):
    aug_X.append(X_tay + rng.normal(0, 0.01, X_tay.shape).astype(np.float32))
    aug_y.append(np.ones(len(X_tay), dtype=np.float32))

X_tay_aug = np.vstack(aug_X)
y_tay_aug = np.hstack(aug_y)
print(f"   Augmented Tay samples: {len(X_tay_aug)}")

# ============ STEP 4: BALANCE ============
n_tay = len(X_tay_aug)
n_other = min(len(X_other), n_tay * 2)
idx = rng.choice(len(X_other), n_other, replace=False)
X = np.vstack([X_tay_aug, X_other[idx]])
y = np.hstack([np.ones(n_tay, dtype=np.float32), np.zeros(n_other, dtype=np.float32)])
perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
print(f"   Total: {len(X)}  Tay: {int(y.sum())}  Other: {len(y)-int(y.sum())}")

# ============ STEP 5: SPLIT ============
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)
print(f"   Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")

cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_tr)
class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
print(f"   Class weights: {class_weight_dict}")

# ============ STEP 6: MODEL ============
print("\n🏗️ Building model...")
model = Sequential([
    Input(shape=(42,)),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.4),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.4),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'), BatchNormalization(), Dropout(0.2),
    Dense(32,  activation='relu'),
    Dense(1,   activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()

# ============ STEP 7: TRAIN ============
print("\n🚀 Training...")
cbs = [
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'tay_best.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
]
model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
          epochs=200, batch_size=32, callbacks=cbs,
          class_weight=class_weight_dict, verbose=1)

# ============ STEP 8: EVALUATE ============
_, te_acc, te_auc = model.evaluate(X_te, y_te, verbose=0)
print(f"\n📊 Test:  Acc={te_acc*100:.2f}%  AUC={te_auc:.4f}")

tay_preds = model.predict(X_tay, verbose=0).flatten()
tay_rate = (tay_preds > THRESHOLD).mean() * 100
print(f"   Original Tay detection: {tay_rate:.1f}%  (avg conf {tay_preds.mean():.4f})")

other_preds = model.predict(X_other, verbose=0).flatten()
fp = (other_preds > THRESHOLD).sum()
print(f"   False positives: {fp}/{len(X_other)}  ({(1-fp/len(X_other))*100:.1f}% specificity)")

# ============ STEP 9: SAVE ============
print("\n💾 Saving...")
model.save(os.path.join(OUTPUT_DIR, 'tay_robust.h5'))

conv = tf.lite.TFLiteConverter.from_keras_model(model)
tfl = conv.convert()
with open(os.path.join(OUTPUT_DIR, 'tay_robust.tflite'), 'wb') as f:
    f.write(tfl)
print(f"   TFLite size: {len(tfl)/1024:.1f} KB")

info = {
    'model': 'Tay Robust v2',
    'letter': TAY_LETTER,
    'preprocessing': 'wrist-relative + wrist->middle_MCP scale + handedness mirror',
    'handedness_seen': hands_seen,
    'test_accuracy': float(te_acc), 'test_auc': float(te_auc),
    'tay_detection_rate': float(tay_rate),
    'threshold': THRESHOLD,
    'classes': ['Not Tay', 'Tay']
}
with open(os.path.join(OUTPUT_DIR, 'tay_robust_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ DONE. Saved to: {OUTPUT_DIR}")