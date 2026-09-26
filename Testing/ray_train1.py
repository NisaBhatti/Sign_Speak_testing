# Save as: train_ray.py
"""
STRICT Ray (ر) trainer.
- Uses hard negatives (signs similar to Ray)
- Auto-tunes threshold with HIGH PRECISION preference
- Saves Keras + TFLite + info.json
"""

import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import json, os, re

# ======================= CONFIG =======================
DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Exported_Model\exported_models_ray"
TABLE_NAME = "rightHandDataset"
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
RAY_LETTER = 'ر'
SEED = 42

# Signs that LOOK LIKE Ray → treat as hard negatives (higher weight)
HARD_NEGATIVE_LETTERS = {'ا', 'أ', 'إ', 'آ', 'ط', 'ظ', 'ن', 'ي', 'ئ', 'ى', 'ل', 'لا'}
# ======================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("TRAIN RAY (ر) — STRICT MODEL")
print("=" * 60)

# ---------- Load DB ----------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
col_names = [c[1] for c in cursor.fetchall()]
n_cols = len(col_names)

label_idx = n_cols - 1
for i, name in enumerate(col_names):
    if name.lower() in ('label', 'letter', 'rletter', 'class', 'sign'):
        label_idx = i
        break
feature_start = 1 if col_names[0].lower() in ('id', 'rowid') else 0
feature_indices = list(range(feature_start, label_idx))[:42]

cursor.execute(f"SELECT * FROM {TABLE_NAME}")
rows = cursor.fetchall()
conn.close()

print(f"\nLoaded {len(rows)} rows")

X_list, y_list = [], []
for row in rows:
    try:
        feats = [float(row[i]) for i in feature_indices]
        label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', str(row[label_idx])).strip()
        X_list.append(feats)
        y_list.append(label)
    except (ValueError, IndexError):
        continue

X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list)
print(f"Usable: {len(X_all)}")

# ---------- Normalize ----------
X_norm = X_all.copy()
X_norm[:, 0::2] = X_all[:, 0::2] / IMAGE_WIDTH
X_norm[:, 1::2] = X_all[:, 1::2] / IMAGE_HEIGHT

# ---------- Split Ray vs other ----------
ray_mask = y_all == RAY_LETTER
X_ray = X_norm[ray_mask]
y_ray_labels = y_all[ray_mask]
X_other = X_norm[~ray_mask]
y_other_labels = y_all[~ray_mask]

print(f"\nRay:   {len(X_ray)}")
print(f"Other: {len(X_other)}")

# Which other classes exist?
from collections import Counter
other_counts = Counter(y_other_labels)
print(f"\nOther classes ({len(other_counts)}):")
for lbl, cnt in sorted(other_counts.items(), key=lambda x: -x[1]):
    mark = " ← HARD" if lbl in HARD_NEGATIVE_LETTERS else ""
    print(f"   '{lbl}': {cnt}{mark}")

if len(X_ray) < 20:
    raise SystemExit("❌ Need ≥20 Ray samples. Run prepare_ray_data.py")

# ---------- Augment Ray (light) ----------
print(f"\n🔧 Light augmentation on Ray...")
aug_X = [X_ray]
aug_y = [np.ones(len(X_ray), dtype=np.float32)]

for sigma in (0.005, 0.01):
    aug_X.append(np.clip(X_ray + np.random.normal(0, sigma, X_ray.shape).astype(np.float32), 0, 1))
    aug_y.append(np.ones(len(X_ray), dtype=np.float32))

for s in (0.97, 1.03):
    aug_X.append(np.clip(X_ray * s, 0, 1))
    aug_y.append(np.ones(len(X_ray), dtype=np.float32))

for dx, dy in ((0.008, 0), (-0.008, 0), (0, 0.008), (0, -0.008)):
    sh = X_ray.copy()
    sh[:, 0::2] = np.clip(sh[:, 0::2] + dx, 0, 1)
    sh[:, 1::2] = np.clip(sh[:, 1::2] + dy, 0, 1)
    aug_X.append(sh)
    aug_y.append(np.ones(len(X_ray), dtype=np.float32))

X_ray_aug = np.vstack(aug_X)
y_ray_aug = np.hstack(aug_y)
print(f"Ray after augmentation: {len(X_ray_aug)}")

# ---------- Build "Other" pool with HARD negatives prioritized ----------
n_ray = len(X_ray_aug)

hard_mask = np.isin(y_other_labels, list(HARD_NEGATIVE_LETTERS))
X_hard = X_other[hard_mask]
X_easy = X_other[~hard_mask]
print(f"\nHard negatives available: {len(X_hard)}")
print(f"Easy negatives available: {len(X_easy)}")

# Compose "other" pool: take ALL hard + fill from easy
if len(X_hard) >= n_ray:
    idx = np.random.choice(len(X_hard), n_ray, replace=False)
    X_other_bal = X_hard[idx]
    print(f"Using {n_ray} HARD negatives only")
else:
    idx_h = np.random.choice(len(X_hard), len(X_hard), replace=False)
    X_hard_used = X_hard[idx_h]
    need = n_ray - len(X_hard_used)
    if len(X_easy) >= need:
        idx_e = np.random.choice(len(X_easy), need, replace=False)
        X_easy_used = X_easy[idx_e]
    else:
        idx_e = np.random.choice(len(X_easy), need, replace=True)
        X_easy_used = X_easy[idx_e]
    X_other_bal = np.vstack([X_hard_used, X_easy_used])
    print(f"Using {len(X_hard_used)} HARD + {len(X_easy_used)} easy negatives")

y_other_bal = np.zeros(len(X_other_bal), dtype=np.float32)

X = np.vstack([X_ray_aug, X_other_bal]).astype(np.float32)
y = np.hstack([y_ray_aug, y_other_bal]).astype(np.float32)

perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]
print(f"\nTotal: {len(X)} | Ray: {int(y.sum())} | Other: {int((y==0).sum())}")

# ---------- Split ----------
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.176, random_state=SEED, stratify=y_tv)
print(f"Split: Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

# ---------- Model ----------
print(f"\n🏗️ Building model...")
model = Sequential([
    Input(shape=(42,)),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.35),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.35),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.30),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.25),
    Dense(64, activation='relu'), BatchNormalization(), Dropout(0.20),
    Dense(32, activation='relu'), Dropout(0.10),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
)
model.summary()

# ---------- Train ----------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_ray.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
]

print(f"\n🚀 Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, batch_size=32,
    callbacks=callbacks, verbose=1,
)

# ---------- Auto-tune threshold (PRECISION-FIRST) ----------
print(f"\n🎯 Auto-tuning threshold (precision-first)...")
val_preds = model.predict(X_val, verbose=0).flatten()

best_t, best_score = 0.5, -1
for t in np.arange(0.3, 0.96, 0.01):
    b = (val_preds > t).astype(int)
    p = precision_score(y_val, b, zero_division=0)
    r = recall_score(y_val, b, zero_division=0)
    if p >= 0.92 and r >= 0.85:          # prefer high-precision thresholds
        score = p * 0.7 + r * 0.3
        if score > best_score:
            best_score = score
            best_t = float(t)

# fallback: max F1
if best_score < 0:
    print("   ⚠️ No high-precision threshold found, using max F1")
    for t in np.arange(0.3, 0.96, 0.01):
        f1 = f1_score(y_val, (val_preds > t).astype(int), zero_division=0)
        if f1 > best_score:
            best_score = f1
            best_t = float(t)

print(f"   Chosen threshold: {best_t:.3f}")

# ---------- Evaluate ----------
def report(name, Xs, ys):
    preds = model.predict(Xs, verbose=0).flatten()
    b = (preds > best_t).astype(int)
    acc = (b == ys).mean()
    p = precision_score(ys, b, zero_division=0)
    r = recall_score(ys, b, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(ys, b, labels=[0, 1]).ravel()
    print(f"   {name:6s}: Acc={acc*100:.2f}%  P={p*100:.1f}%  R={r*100:.1f}%  FP={fp} FN={fn}")
    return preds

print(f"\n📊 Metrics (threshold={best_t:.3f}):")
report("Train", X_train, y_train)
report("Val",   X_val,   y_val)
report("Test",  X_test,  y_test)

# Original Ray
orig = model.predict(X_ray, verbose=0).flatten()
det = (orig > best_t).sum()
print(f"\n   Original Ray: {det}/{len(X_ray)} ({det/len(X_ray)*100:.1f}%)")
print(f"   Avg conf: {orig.mean():.4f} ({orig.mean()*100:.1f}%)")
print(f"   Min conf: {orig.min():.4f}")

# Evaluate on EACH other class separately
print(f"\n   Per-class false-positive rate:")
fp_report = {}
for lbl in sorted(other_counts.keys()):
    mask = y_other_labels == lbl
    if mask.sum() == 0:
        continue
    cls_X = X_other[mask]
    preds = model.predict(cls_X, verbose=0).flatten()
    fp = (preds > best_t).sum()
    rate = fp / len(cls_X)
    fp_report[lbl] = float(rate)
    flag = " ← HARD" if lbl in HARD_NEGATIVE_LETTERS else ""
    if rate > 0.05 or lbl in HARD_NEGATIVE_LETTERS:
        print(f"      '{lbl}': {fp}/{len(cls_X)} ({rate*100:.1f}%){flag}")

# ---------- Save ----------
print(f"\n💾 Saving...")
model.save(os.path.join(OUTPUT_DIR, 'ray_robust.h5'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
tflite = converter.convert()
with open(os.path.join(OUTPUT_DIR, 'ray_robust.tflite'), 'wb') as f:
    f.write(tflite)

info = {
    "model": "Ray Robust (strict)",
    "letter": RAY_LETTER,
    "threshold": best_t,
    "hard_negatives": sorted(list(HARD_NEGATIVE_LETTERS)),
    "ray_samples": int(len(X_ray)),
    "avg_ray_confidence": float(orig.mean()),
    "min_ray_confidence": float(orig.min()),
    "fp_per_class": fp_report,
    "image_size": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
}
with open(os.path.join(OUTPUT_DIR, 'ray_robust_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ DONE. Threshold={best_t:.3f}")
print(f"   Next: python test_ray.py")