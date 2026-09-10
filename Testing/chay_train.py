# Save as: D:\MODEL\Sign_Speak_testing-main\Testing\chay_train.py
# Chay (چ) Robust Training — v3 (diagnostic + fixed)

import os, cv2, json, numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tensorflow.keras import layers, models, callbacks, optimizers
from collections import Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 60)
print("SIGNSPEAK - Chay Robust Training (v3)")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================
CHAY_DIR   = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Chay"
OUTPUT_DIR = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFLITE_PATH = os.path.join(OUTPUT_DIR, "chay_robust.tflite")
KERAS_PATH  = os.path.join(OUTPUT_DIR, "chay_robust.keras")
INFO_PATH   = os.path.join(OUTPUT_DIR, "chay_robust_info.json")
DEBUG_DIR   = os.path.join(OUTPUT_DIR, "chay_debug_landmarks")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
VALID_EXT     = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUG_PER_IMAGE = 40
NEG_PER_POS   = 2
EPOCHS        = 120
BATCH_SIZE    = 32
SEED          = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# VERIFY DATASET
# ============================================================
if not os.path.isdir(CHAY_DIR):
    raise FileNotFoundError(f"Chay folder not found: {CHAY_DIR}")

chay_files = [f for f in os.listdir(CHAY_DIR) if f.lower().endswith(VALID_EXT)]
print(f"\n[INFO] Found {len(chay_files)} Chay images")

if len(chay_files) < 10:
    raise ValueError("Need at least 10 Chay images.")

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

# ============================================================
# NORMALIZATION (wrist-centered, unit-scaled)
# ============================================================
def normalize_landmarks(feat):
    pts = feat.reshape(21, 2).copy()
    pts -= pts[0]
    scale = np.max(np.abs(pts))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.reshape(-1).astype(np.float32)

# ============================================================
# LOAD POSITIVES + DIAGNOSTIC
# ============================================================
print("\n[INFO] Extracting landmarks from Chay images...")
raw_pos = []
skipped = 0
skipped_files = []

for i, fname in enumerate(chay_files):
    img = cv2.imread(os.path.join(CHAY_DIR, fname))
    if img is None:
        skipped += 1
        skipped_files.append(fname)
        continue
    feats = extract_landmarks(img)
    if feats is None:
        skipped += 1
        skipped_files.append(fname)
        continue
    raw_pos.append(normalize_landmarks(feats))
    if (i + 1) % 25 == 0:
        print(f"   processed {i + 1}/{len(chay_files)}")

pos_features = np.array(raw_pos, dtype=np.float32)
print(f"\n[INFO] Valid Chay samples: {len(pos_features)} (skipped {skipped})")

if skipped_files:
    print(f"[INFO] Skipped files (no hand detected by MediaPipe):")
    for f in skipped_files[:10]:
        print(f"       - {f}")
    if len(skipped_files) > 10:
        print(f"       ... and {len(skipped_files) - 10} more")

if len(pos_features) < 10:
    raise ValueError("Not enough valid Chay landmarks.")

# ============================================================
# 🔍 CONSISTENCY DIAGNOSTIC — most important step
#    Measures how similar your Chay images are to each other.
#    If the average distance between samples is HIGH, your dataset
#    is inconsistent → low accuracy is a DATA problem.
# ============================================================
print("\n[DIAGNOSTIC] Measuring dataset consistency...")

# pairwise L2 distances
n = len(pos_features)
dists = []
for i in range(n):
    for j in range(i + 1, n):
        d = np.linalg.norm(pos_features[i] - pos_features[j])
        dists.append(d)

dists = np.array(dists)
mean_d = float(dists.mean())
std_d  = float(dists.std())
max_d  = float(dists.max())
min_d  = float(dists.min())

print(f"   Mean pairwise distance : {mean_d:.4f}")
print(f"   Std                    : {std_d:.4f}")
print(f"   Min / Max              : {min_d:.4f} / {max_d:.4f}")

# Interpretation
if mean_d < 0.30:
    consistency = "EXCELLENT — poses are very similar"
elif mean_d < 0.60:
    consistency = "GOOD — normal variation"
elif mean_d < 1.00:
    consistency = "POOR — poses vary a lot"
else:
    consistency = "BAD — poses are inconsistent (this is why accuracy is low)"

print(f"   → Consistency: {consistency}")

# Save per-sample "outlier score" — how far each sample is from the mean pose
mean_pose = pos_features.mean(axis=0)
outlier_scores = np.linalg.norm(pos_features - mean_pose, axis=1)

print(f"\n[DIAGNOSTIC] Outlier scores (higher = more different from average):")
order = np.argsort(outlier_scores)[::-1]
for rank, idx in enumerate(order[:10], 1):
    src = [f for f in chay_files if f not in skipped_files]
    label = src[idx] if idx < len(src) else f"idx_{idx}"
    print(f"   {rank:2d}. score={outlier_scores[idx]:.4f}   {label}")

# Save a visualization of the mean pose
def draw_landmarks_on_canvas(pts_2d, size=400, color=(0, 255, 0), thickness=2):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    # Map [-1,1] to [0,size]
    def to_px(p):
        x = int((p[0] + 1) / 2 * (size - 1))
        y = int((p[1] + 1) / 2 * (size - 1))
        return x, y
    for c in mp_hands.HAND_CONNECTIONS:
        cv2.line(canvas, to_px(pts_2d[c[0]]), to_px(pts_2d[c[1]]), color, thickness)
    for p in pts_2d:
        cv2.circle(canvas, to_px(p), 5, (255, 255, 255), -1)
    return canvas

mean_canvas = draw_landmarks_on_canvas(mean_pose.reshape(21, 2))
cv2.imwrite(os.path.join(DEBUG_DIR, "mean_chay_pose.png"), mean_canvas)

# Save the 5 most different samples as images
for rank, idx in enumerate(order[:5], 1):
    canvas = draw_landmarks_on_canvas(pos_features[idx].reshape(21, 2),
                                      color=(0, 165, 255))
    cv2.imwrite(os.path.join(DEBUG_DIR, f"outlier_{rank}_score_{outlier_scores[idx]:.2f}.png"),
                canvas)

print(f"[INFO] Debug images saved to: {DEBUG_DIR}")

# ============================================================
# OPTIONAL: DROP EXTREME OUTLIERS
#    If your consistency is POOR/BAD, drop the top 20% outliers
#    so the model learns the *actual* Chay pose, not noise.
# ============================================================
if mean_d >= 0.60:
    drop_n = max(1, int(len(pos_features) * 0.20))
    keep_idx = np.argsort(outlier_scores)[:-drop_n]
    pos_features = pos_features[keep_idx]
    print(f"\n[INFO] Dropped {drop_n} outlier samples (dataset was inconsistent).")
    print(f"[INFO] Remaining Chay samples: {len(pos_features)}")

# ============================================================
# STRONG AUGMENTATION
# ============================================================
def augment_landmarks(feat):
    pts = feat.reshape(21, 2).copy()

    angle = np.deg2rad(np.random.uniform(-35, 35))
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    pts = pts @ R.T

    pts *= np.random.uniform(0.88, 1.12)
    pts += np.random.uniform(-0.08, 0.08, size=(1, 2))
    pts += np.random.normal(0, 0.008, pts.shape)

    return pts.reshape(-1).astype(np.float32)

aug_pos = []
for f in pos_features:
    aug_pos.append(f)
    for _ in range(AUG_PER_IMAGE):
        aug_pos.append(augment_landmarks(f))
aug_pos = np.array(aug_pos, dtype=np.float32)
print(f"\n[INFO] Augmented Chay samples: {len(aug_pos)}")

# ============================================================
# REALISTIC HARD NEGATIVES (non-Chay hand shapes)
# ============================================================
print("\n[INFO] Generating hard negatives...")

def random_hand_shape():
    pts = [np.array([0.0, 0.0])]
    # 4 fingers × 4 points
    for _ in range(4):
        base_angle = np.deg2rad(np.random.uniform(-90, 90))
        direction = np.array([np.sin(base_angle), -np.cos(base_angle)])
        length = np.random.uniform(0.4, 1.0)
        step = direction * (length / 4.0)
        p = direction * 0.15
        for _j in range(4):
            p = p + step + np.random.normal(0, 0.02, 2)
            pts.append(p)
    # Thumb
    base_angle = np.deg2rad(np.random.uniform(20, 90))
    direction = np.array([np.cos(base_angle), np.sin(base_angle)])
    p = direction * 0.15
    step = direction * (np.random.uniform(0.3, 0.7) / 4.0)
    for _j in range(4):
        p = p + step + np.random.normal(0, 0.02, 2)
        pts.append(p)
    pts = np.array(pts, dtype=np.float32)
    return normalize_landmarks(pts.reshape(-1))

def distorted_chay(feat):
    pts = feat.reshape(21, 2).copy()
    pts *= np.random.uniform(0.4, 0.6)
    angle = np.deg2rad(np.random.uniform(60, 120))
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    pts = pts @ R.T
    pts += np.random.uniform(-0.4, 0.4, size=(1, 2))
    return normalize_landmarks(pts.reshape(-1))

n_neg = int(len(aug_pos) * NEG_PER_POS)
neg_features = []
for _ in range(n_neg):
    r = np.random.rand()
    if r < 0.7:
        neg_features.append(random_hand_shape())
    else:
        idx = np.random.randint(0, len(pos_features))
        neg_features.append(distorted_chay(pos_features[idx]))
neg_features = np.array(neg_features, dtype=np.float32)
print(f"[INFO] Negative samples: {len(neg_features)}")

# ============================================================
# ASSEMBLE
# ============================================================
X = np.vstack([aug_pos, neg_features]).astype(np.float32)
y = np.hstack([
    np.ones(len(aug_pos), dtype=np.float32),
    np.zeros(len(neg_features), dtype=np.float32)
])
print(f"\n[INFO] Dataset: {len(X)} samples "
      f"({int(y.sum())} pos / {int((y == 0).sum())} neg)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"[INFO] Train: {len(X_train)}  Test: {len(X_test)}")

# ============================================================
# MODEL
# ============================================================
def build_model(input_dim=42):
    inp = layers.Input(shape=(input_dim,), name="landmarks")

    x = layers.Dense(256)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid", name="chay_score")(x)

    model = models.Model(inp, out, name="chay_robust")
    model.compile(
        optimizer=optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

model = build_model()
model.summary()

# ============================================================
# CLASS WEIGHTS
# ============================================================
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
# EVALUATE
# ============================================================
print("\n[INFO] Evaluating...")
y_prob = model.predict(X_test, verbose=0).ravel()

best_thr, best_score = 0.5, -1.0
for thr in np.linspace(0.05, 0.95, 91):
    preds = (y_prob > thr).astype(int)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    fn = np.sum((preds == 0) & (y_test == 1))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    score = f1 * 0.6 + prec * 0.4
    if score > best_score:
        best_score, best_thr = score, thr

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
    "model": "chay_robust",
    "sign": "چ",
    "sign_name": "Chay",
    "input_shape": [42],
    "input_format": "normalized (x,y) 21 landmarks",
    "preprocessing": "wrist-centered + unit-scaled",
    "threshold": float(round(best_thr, 3)),
    "test_accuracy": float(round(acc, 4)),
    "test_auc": float(round(auc, 4)),
    "test_precision": float(round(precision, 4)),
    "test_recall": float(round(recall, 4)),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "dataset_consistency": {
        "mean_distance": mean_d,
        "std_distance":  std_d,
        "verdict": consistency
    },
    "n_chay_images": len(chay_files),
    "n_valid_chay": int(len(pos_features)),
    "n_augmented_chay": int(len(aug_pos)),
    "n_negatives": int(len(neg_features)),
    "mediapipe": "solutions.hands, max_num_hands=1",
}
with open(INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved:")
print(f"   {TFLITE_PATH}")
print(f"   {KERAS_PATH}")
print(f"   {INFO_PATH}")
print(f"   {DEBUG_DIR}\\ (mean pose + outlier visualizations)")

hands.close()
print("\n" + "=" * 60)
print("DONE. Now run: python test_chay_robust.py")
print("=" * 60)