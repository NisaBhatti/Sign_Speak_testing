# Save as: D:\MODEL\Sign_Speak_testing-main\Testing\jeem_train.py

import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tensorflow.keras import layers, models, callbacks, optimizers

# ============================================================
# SILENCE TF / MEDIAPIPE NOISE (optional, cleaner logs)
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 60)
print("SIGNSPEAK - Jeem Robust Training")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================
JEEM_DIR   = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Jeem"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFLITE_PATH = os.path.join(OUTPUT_DIR, "jeem_robust.tflite")
KERAS_PATH  = os.path.join(OUTPUT_DIR, "jeem_robust.keras")
INFO_PATH   = os.path.join(OUTPUT_DIR, "jeem_robust_info.json")

# ============================================================
# CONFIG
# ============================================================
VALID_EXT       = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NEGATIVE_RATIO  = 3.0      # 3 negatives per positive
JITTER_COPIES   = 2        # extra jittered copies per positive
EPOCHS          = 40
BATCH_SIZE      = 32
SEED            = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. VERIFY DATASET
# ============================================================
if not os.path.isdir(JEEM_DIR):
    raise FileNotFoundError(f"Jeem folder not found: {JEEM_DIR}")

jeem_files = [f for f in os.listdir(JEEM_DIR) if f.lower().endswith(VALID_EXT)]
print(f"\n[INFO] Found {len(jeem_files)} Jeem images in {JEEM_DIR}")

if len(jeem_files) < 10:
    raise ValueError("Need at least 10 Jeem images to train.")

# ============================================================
# 2. MEDIAPIPE HAND LANDMARK EXTRACTOR
# ============================================================
print("\n[INFO] Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def extract_landmarks(image_bgr):
    """Return a 42-dim (x,y) landmark vector normalized 0-1, or None."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    feats = []
    for p in lm:
        feats.extend([p.x, p.y])
    return np.array(feats, dtype=np.float32)

# ============================================================
# 3. LOAD JEEM (POSITIVE) SAMPLES
# ============================================================
print("\n[INFO] Extracting landmarks from Jeem images...")
pos_features = []
skipped = 0

for i, fname in enumerate(jeem_files):
    path = os.path.join(JEEM_DIR, fname)
    img = cv2.imread(path)
    if img is None:
        skipped += 1
        continue
    feats = extract_landmarks(img)
    if feats is None:
        skipped += 1
        continue
    pos_features.append(feats)
    if (i + 1) % 25 == 0:
        print(f"   processed {i + 1}/{len(jeem_files)}")

pos_features = np.array(pos_features, dtype=np.float32)
print(f"[INFO] Valid Jeem samples: {len(pos_features)} (skipped {skipped})")

if len(pos_features) < 10:
    raise ValueError("Not enough valid Jeem landmarks. Check images contain a hand.")

# ============================================================
# 4. AUGMENT JEEM (jitter to simulate natural variation)
# ============================================================
def jitter(feat, sigma=0.02):
    noise = np.random.normal(0, sigma, feat.shape).astype(np.float32)
    out = feat + noise
    return np.clip(out, 0.0, 1.0).astype(np.float32)

aug_pos = []
for f in pos_features:
    aug_pos.append(f)
    for _ in range(JITTER_COPIES):
        aug_pos.append(jitter(f))
aug_pos = np.array(aug_pos, dtype=np.float32)
print(f"[INFO] Augmented Jeem samples: {len(aug_pos)}")

# ============================================================
# 5. GENERATE HARD NEGATIVES (non-Jeem samples)
# ============================================================
print("\n[INFO] Generating negative samples...")

def random_hand_like():
    """Random hand-like landmark cloud — not Jeem."""
    cx = np.random.uniform(0.3, 0.7)
    cy = np.random.uniform(0.3, 0.7)
    pts = []
    for _ in range(21):
        x = np.clip(cx + np.random.normal(0, 0.15), 0.0, 1.0)
        y = np.clip(cy + np.random.normal(0, 0.15), 0.0, 1.0)
        pts.extend([x, y])
    return np.array(pts, dtype=np.float32)

def random_noise():
    return np.random.uniform(0.0, 1.0, 42).astype(np.float32)

def shifted_jeem(feat, shift=0.25):
    """Rotated + shifted Jeem → distorted, treated as non-Jeem."""
    out = feat.copy().reshape(21, 2)
    angle = np.random.uniform(-np.pi / 2, np.pi / 2)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    out = out @ R.T
    out[:, 0] += np.random.uniform(-shift, shift)
    out[:, 1] += np.random.uniform(-shift, shift)
    out = np.clip(out, 0.0, 1.0).reshape(-1)
    return out.astype(np.float32)

n_neg = int(len(aug_pos) * NEGATIVE_RATIO)
neg_features = []

for _ in range(n_neg):
    r = np.random.rand()
    if r < 0.5:
        neg_features.append(random_hand_like())
    elif r < 0.8:
        neg_features.append(random_noise())
    else:
        # ✅ FIX: pick a random ROW index, not the whole 2D array
        idx = np.random.randint(0, len(pos_features))
        neg_features.append(shifted_jeem(pos_features[idx]))

neg_features = np.array(neg_features, dtype=np.float32)
print(f"[INFO] Negative samples: {len(neg_features)}")

# ============================================================
# 6. ASSEMBLE DATASET
# ============================================================
X = np.vstack([aug_pos, neg_features]).astype(np.float32)
y = np.hstack([
    np.ones(len(aug_pos),  dtype=np.float32),
    np.zeros(len(neg_features), dtype=np.float32)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[INFO] Train: {len(X_train)}  Test: {len(X_test)}")

# ============================================================
# 7. BUILD MODEL
# ============================================================
def build_model(input_dim=42):
    inp = layers.Input(shape=(input_dim,), name="landmarks")

    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid", name="jeem_score")(x)

    model = models.Model(inp, out, name="jeem_robust")
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

model = build_model()
model.summary()

# ============================================================
# 8. TRAIN
# ============================================================
cb = [
    callbacks.EarlyStopping(monitor="val_auc", patience=8, mode="max",
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=4, min_lr=1e-5),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=1
)

# ============================================================
# 9. EVALUATE + PICK BEST THRESHOLD
# ============================================================
print("\n[INFO] Evaluating...")
y_prob = model.predict(X_test, verbose=0).ravel()

best_thr = 0.5
best_score = -1.0

for thr in np.linspace(0.05, 0.95, 91):
    preds = (y_prob > thr).astype(int)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    # Favor precision (detect Jeem and nothing else)
    score = f1 * 0.6 + prec * 0.4
    if score > best_score:
        best_score = score
        best_thr   = thr

final_preds = (y_prob > best_thr).astype(int)
acc = accuracy_score(y_test, final_preds)
auc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, final_preds)

# Safe unpack in case of edge cases
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0

precision = tp / (tp + fp + 1e-9)
recall    = tp / (tp + fn + 1e-9)

print(f"\n[RESULTS]")
print(f"   Threshold: {best_thr:.3f}")
print(f"   Accuracy : {acc * 100:.2f}%")
print(f"   AUC      : {auc:.4f}")
print(f"   Precision: {precision * 100:.2f}%")
print(f"   Recall   : {recall * 100:.2f}%")
print(f"   Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

# ============================================================
# 10. EXPORT TO TFLITE
# ============================================================
print("\n[INFO] Exporting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

model.save(KERAS_PATH)

# ============================================================
# 11. SAVE METADATA
# ============================================================
info = {
    "model": "jeem_robust",
    "sign": "ج",
    "sign_name": "Jeem",
    "input_shape": [42],
    "input_format": "(x,y) for 21 hand landmarks, normalized 0-1",
    "threshold": float(round(best_thr, 3)),
    "test_accuracy": float(round(acc, 4)),
    "test_auc": float(round(auc, 4)),
    "test_precision": float(round(precision, 4)),
    "test_recall": float(round(recall, 4)),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                         "fn": int(fn), "tp": int(tp)},
    "n_jeem_images": len(jeem_files),
    "n_valid_jeem": int(len(pos_features)),
    "n_augmented_jeem": int(len(aug_pos)),
    "n_negatives": int(len(neg_features)),
    "mediapipe": "solutions.hands, max_num_hands=1",
}

with open(INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved:")
print(f"   {TFLITE_PATH}")
print(f"   {KERAS_PATH}")
print(f"   {INFO_PATH}")

hands.close()

print("\n" + "=" * 60)
print("DONE. Now run: python test_jeem_robust.py")
print("=" * 60)