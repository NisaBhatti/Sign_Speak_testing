# Save as: D:\MODEL\Sign_Speak_testing-main\Testing\tay_train.py
# Tay (ط) Robust Training — v5 (explicit Alif-hard-negatives + auto-threshold)

import os, cv2, json, numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tensorflow.keras import layers, models, callbacks, optimizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 60)
print("SIGNSPEAK - Tay Robust Training (v5)")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================
DATASET_ROOT = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset"
TAY_DIR      = os.path.join(DATASET_ROOT, "Tay")
OUTPUT_DIR   = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFLITE_PATH = os.path.join(OUTPUT_DIR, "tay_robust.tflite")
KERAS_PATH  = os.path.join(OUTPUT_DIR, "tay_robust.keras")
INFO_PATH   = os.path.join(OUTPUT_DIR, "tay_robust_info.json")

# ============================================================
# CONFIG
# ============================================================
VALID_EXT      = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUG_PER_IMAGE  = 30
EXTRA_NEG_MULT = 0.5
EPOCHS         = 120
BATCH_SIZE     = 32
SEED           = 42

# Signs to explicitly treat as "hard" negatives (these confuse Tay the most)
HARD_NEGATIVE_SIGNS = {"alif", "jeem", "chay", "baa", "bay"}

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# VERIFY
# ============================================================
if not os.path.isdir(TAY_DIR):
    raise FileNotFoundError(f"Tay folder not found: {TAY_DIR}")

tay_files = [f for f in os.listdir(TAY_DIR) if f.lower().endswith(VALID_EXT)]
print(f"\n[INFO] Found {len(tay_files)} Tay images")

# Discover other sign folders (all except Tay)
other_signs = []
for name in sorted(os.listdir(DATASET_ROOT)):
    full = os.path.join(DATASET_ROOT, name)
    if not os.path.isdir(full):
        continue
    if name.lower() == "tay":
        continue
    imgs = [f for f in os.listdir(full) if f.lower().endswith(VALID_EXT)]
    if len(imgs) > 0:
        is_hard = name.lower() in HARD_NEGATIVE_SIGNS
        other_signs.append((name, full, imgs, is_hard))

print(f"\n[INFO] Sign folders used as negatives:")
for name, _, imgs, is_hard in other_signs:
    tag = " ★ HARD NEGATIVE" if is_hard else ""
    print(f"       - {name:15s} ({len(imgs)} images){tag}")

if not any(name.lower() == "alif" for name, _, _, _ in other_signs):
    print("\n⚠ WARNING: No Alif folder found under Simple_Dataset.")
    print("   The model will keep confusing Alif as Tay.")
    print("   → Create a folder called 'Alif' next to 'Tay' and add Alif images.")

# ============================================================
# MEDIAPIPE
# ============================================================
print("\n[INFO] Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.2
)

def extract_landmarks(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    feats = []
    for p in lm:
        feats.extend([p.x, p.y])
    return np.array(feats, dtype=np.float32)

def normalize_landmarks(feat):
    pts = feat.reshape(21, 2).copy()
    pts -= pts[0]
    scale = np.max(np.abs(pts))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.reshape(-1).astype(np.float32)

# ============================================================
# LOAD TAY (POSITIVE)
# ============================================================
print("\n[INFO] Extracting Tay landmarks...")
pos_features = []
skipped = 0
for i, fname in enumerate(tay_files):
    img = cv2.imread(os.path.join(TAY_DIR, fname))
    if img is None:
        skipped += 1; continue
    feats = extract_landmarks(img)
    if feats is None:
        skipped += 1; continue
    pos_features.append(normalize_landmarks(feats))
    if (i + 1) % 25 == 0:
        print(f"   processed {i + 1}/{len(tay_files)}")

pos_features = np.array(pos_features, dtype=np.float32)
print(f"[INFO] Valid Tay samples: {len(pos_features)} (skipped {skipped})")

if len(pos_features) < 10:
    raise ValueError("Not enough valid Tay landmarks.")

# ============================================================
# LOAD NEGATIVES FROM OTHER SIGNS (with per-sign tracking)
# ============================================================
print("\n[INFO] Extracting landmarks from other signs...")
neg_by_sign = {}   # sign_name → list of feature vectors

for sign_name, sign_dir, files, is_hard in other_signs:
    feats_list = []
    for fname in files:
        img = cv2.imread(os.path.join(sign_dir, fname))
        if img is None:
            continue
        feats = extract_landmarks(img)
        if feats is None:
            continue
        feats_list.append(normalize_landmarks(feats))
    neg_by_sign[sign_name] = np.array(feats_list, dtype=np.float32)
    tag = " (hard)" if is_hard else ""
    print(f"   {sign_name:15s} → {len(feats_list)} valid landmarks{tag}")

all_real_neg = np.vstack([v for v in neg_by_sign.values() if len(v) > 0]) \
    if any(len(v) > 0 for v in neg_by_sign.values()) \
    else np.zeros((0, 42), dtype=np.float32)
print(f"[INFO] Total REAL negatives: {len(all_real_neg)}")

# ============================================================
# SEPARATION CHECK — Tay vs each sign
# ============================================================
print("\n[DIAGNOSTIC] Distance from Tay mean to each sign's mean:")
tay_mean = pos_features.mean(axis=0)
for sign_name, feats in neg_by_sign.items():
    if len(feats) == 0:
        continue
    d = float(np.linalg.norm(tay_mean - feats.mean(axis=0)))
    verdict = "  ← VERY CLOSE (hard to separate)" if d < 0.35 else ""
    print(f"   {sign_name:15s} : {d:.4f}{verdict}")

# ============================================================
# AUGMENT POSITIVES
# ============================================================
def augment_landmarks(feat, strength=1.0):
    pts = feat.reshape(21, 2).copy()
    angle = np.deg2rad(np.random.uniform(-35, 35) * strength)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    pts = pts @ R.T
    pts *= np.random.uniform(1 - 0.12 * strength, 1 + 0.12 * strength)
    pts += np.random.uniform(-0.08 * strength, 0.08 * strength, size=(1, 2))
    pts += np.random.normal(0, 0.008 * strength, pts.shape)
    return pts.reshape(-1).astype(np.float32)

aug_pos = []
for f in pos_features:
    aug_pos.append(f)
    for _ in range(AUG_PER_IMAGE):
        aug_pos.append(augment_landmarks(f, 1.0))
aug_pos = np.array(aug_pos, dtype=np.float32)
print(f"\n[INFO] Augmented Tay samples: {len(aug_pos)}")

# ============================================================
# AUGMENT NEGATIVES — extra emphasis on HARD signs (Alif etc.)
# ============================================================
aug_neg = []
for sign_name, feats in neg_by_sign.items():
    if len(feats) == 0:
        continue
    is_hard = sign_name.lower() in HARD_NEGATIVE_SIGNS
    copies = 8 if is_hard else 3         # ← hard negatives get MORE copies
    for f in feats:
        aug_neg.append(f)
        for _ in range(copies):
            # slight stronger aug on hard negs to force separation
            aug_neg.append(augment_landmarks(f, 1.2 if is_hard else 1.0))

aug_neg = np.array(aug_neg, dtype=np.float32)
print(f"[INFO] Augmented negatives: {len(aug_neg)}")

# ============================================================
# EXTRA SYNTHETIC NEGATIVES
# ============================================================
def random_hand_shape():
    pts = [np.array([0.0, 0.0])]
    for _ in range(4):
        base_angle = np.deg2rad(np.random.uniform(-90, 90))
        direction = np.array([np.sin(base_angle), -np.cos(base_angle)])
        length = np.random.uniform(0.4, 1.0)
        step = direction * (length / 4.0)
        p = direction * 0.15
        for _j in range(4):
            p = p + step + np.random.normal(0, 0.02, 2)
            pts.append(p)
    base_angle = np.deg2rad(np.random.uniform(20, 90))
    direction = np.array([np.cos(base_angle), np.sin(base_angle)])
    p = direction * 0.15
    step = direction * (np.random.uniform(0.3, 0.7) / 4.0)
    for _j in range(4):
        p = p + step + np.random.normal(0, 0.02, 2)
        pts.append(p)
    return normalize_landmarks(np.array(pts, dtype=np.float32).reshape(-1))

n_extra = int(len(aug_pos) * EXTRA_NEG_MULT)
extra_neg = np.array([random_hand_shape() for _ in range(n_extra)],
                     dtype=np.float32)
print(f"[INFO] Extra synthetic negatives: {len(extra_neg)}")

neg_features = np.vstack([aug_neg, extra_neg]).astype(np.float32)
print(f"[INFO] Total negatives: {len(neg_features)}")

# cap negatives to 3× positives
if len(neg_features) > 3 * len(aug_pos):
    idx = np.random.permutation(len(neg_features))[: 3 * len(aug_pos)]
    neg_features = neg_features[idx]
    print(f"[INFO] Downsampled negatives to: {len(neg_features)}")

# ============================================================
# ASSEMBLE
# ============================================================
X = np.vstack([aug_pos, neg_features]).astype(np.float32)
y = np.hstack([np.ones(len(aug_pos), dtype=np.float32),
               np.zeros(len(neg_features), dtype=np.float32)])
print(f"\n[INFO] Dataset: {len(X)} samples "
      f"({int(y.sum())} pos / {int((y==0).sum())} neg)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"[INFO] Train: {len(X_train)}  Test: {len(X_test)}")

# ============================================================
# MODEL
# ============================================================
def build_model(input_dim=42):
    inp = layers.Input(shape=(input_dim,), name="landmarks")

    x = layers.Dense(256)(inp); x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x); x = layers.Dropout(0.4)(x)

    x = layers.Dense(128)(x); x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x); x = layers.Dropout(0.4)(x)

    x = layers.Dense(64)(x); x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x); x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x); x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid", name="tay_score")(x)

    model = models.Model(inp, out, name="tay_robust")
    model.compile(
        optimizer=optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

model = build_model()
model.summary()

cw = {0: 1.0, 1: float((y_train == 0).sum() / max(1, (y_train == 1).sum()))}
print(f"[INFO] Class weights: {cw}")

# ============================================================
# TRAIN
# ============================================================
cb = [
    callbacks.EarlyStopping(monitor="val_auc", patience=20, mode="max",
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=8, min_lr=1e-6),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    class_weight=cw,
    verbose=1
)

# ============================================================
# PER-SIGN FALSE-POSITIVE ANALYSIS  ← KEY STEP
# ============================================================
print("\n[ANALYSIS] Running each sign through the trained model...")

def score_features(feats_batch):
    return model.predict(feats_batch, verbose=0).ravel()

# Tay scores
tay_scores = score_features(pos_features)
print(f"\n   Tay       : mean={tay_scores.mean():.3f}  "
      f"min={tay_scores.min():.3f}  max={tay_scores.max():.3f}")

sign_scores = {}
for sign_name, feats in neg_by_sign.items():
    if len(feats) == 0:
        continue
    s = score_features(feats)
    sign_scores[sign_name] = s
    warn = ""
    if s.max() > 0.5:
        warn = "  ⚠ some samples above 0.5!"
    print(f"   {sign_name:9s} : mean={s.mean():.3f}  "
          f"min={s.min():.3f}  max={s.max():.3f}{warn}")

# ============================================================
# AUTO-PICK THRESHOLD that rejects EVERY non-Tay sign
# ============================================================
print("\n[INFO] Choosing threshold to reject all non-Tay signs...")

all_neg_max = 0.0
for sign_name, s in sign_scores.items():
    all_neg_max = max(all_neg_max, float(s.max()))
tay_min = float(tay_scores.min())

print(f"   Highest non-Tay score : {all_neg_max:.3f}")
print(f"   Lowest Tay score      : {tay_min:.3f}")

# Candidate threshold: between the two, biased toward rejecting negatives
if tay_min > all_neg_max:
    best_thr = (tay_min + all_neg_max) / 2
    print(f"   → Clean separation! Threshold = {best_thr:.3f}")
else:
    # Ranges overlap → must bias toward precision
    best_thr = max(all_neg_max + 0.05, 0.75)
    print(f"   ⚠ Overlap between Tay and non-Tay scores.")
    print(f"   → Biasing toward precision. Threshold = {best_thr:.3f}")
    print(f"   → Some real Tay signs may be missed. Add more Tay samples if so.")

best_thr = float(np.clip(best_thr, 0.5, 0.95))

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n[INFO] Evaluating on held-out test set...")
y_prob = model.predict(X_test, verbose=0).ravel()
final_preds = (y_prob > best_thr).astype(int)

acc = accuracy_score(y_test, final_preds)
auc = roc_auc_score(y_test, y_prob)
cm  = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
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
# EXPORT
# ============================================================
print("\n[INFO] Exporting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open(TFLITE_PATH, "wb") as f:
    f.write(converter.convert())
model.save(KERAS_PATH)

info = {
    "model": "tay_robust",
    "sign": "ط",
    "sign_name": "Tay",
    "input_shape": [42],
    "input_format": "normalized (x,y) 21 landmarks",
    "preprocessing": "wrist-centered + unit-scaled",
    "threshold": float(round(best_thr, 3)),
    "test_accuracy": float(round(acc, 4)),
    "test_auc": float(round(auc, 4)),
    "test_precision": float(round(precision, 4)),
    "test_recall": float(round(recall, 4)),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                         "fn": int(fn), "tp": int(tp)},
    "per_sign_scores": {
        name: {
            "mean": float(s.mean()),
            "min":  float(s.min()),
            "max":  float(s.max())
        }
        for name, s in sign_scores.items()
    },
    "negatives_per_sign": {k: int(len(v)) for k, v in neg_by_sign.items()},
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
print("DONE. Now run: python test_tay_robust.py")
print("=" * 60)