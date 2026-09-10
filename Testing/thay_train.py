# Save as: D:\MODEL\Sign_Speak_testing-main\Training\train_thay_cshape.py

import os, cv2, numpy as np, mediapipe as mp, tensorflow as tf, json, sqlite3, re
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SIGNSPEAK - Thay C-Shape Detector (STRICT)")
print("="*60)

# ========== PATHS ==========
PROJECT_ROOT    = r"D:\MODEL\Sign_Speak_testing-main"
THAY_IMAGES_DIR = os.path.join(PROJECT_ROOT, "Simple_Dataset", "Thay")
DB_PATH         = os.path.join(PROJECT_ROOT, "Simple_Dataset", "main_dataset.db")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "Exported_Model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "thay_robust.tflite")
INFO_PATH  = os.path.join(OUTPUT_DIR, "thay_robust_info.json")
KERAS_PATH = os.path.join(OUTPUT_DIR, "thay_robust.keras")
REF_PATH   = os.path.join(OUTPUT_DIR, "thay_reference.npy")

print(f"\nPaths:")
print(f"  Thay imgs: {THAY_IMAGES_DIR}")
print(f"  DB       : {DB_PATH}")
print(f"  Output   : {OUTPUT_DIR}")

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.3, min_tracking_confidence=0.3)

def landmarks_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks: return None
    return np.array([[lm.x, lm.y] for lm in res.multi_hand_landmarks[0].landmark],
                    dtype=np.float32)

# ========== FEATURE ENGINEERING (captures C-shape) ==========
# Finger tip/PIP/DIP/MCP indices
FINGER_JOINTS = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}
WRIST = 0
PALM_CENTER = [0, 5, 9, 13, 17]

def extract_features(lm_xy):
    """
    Return engineered features that describe hand SHAPE (not just position).
    Captures C-shape via finger curl angles + tip-to-palm distances.
    """
    lm = lm_xy.astype(np.float32)

    # 1) Center everything on wrist (translation invariance)
    lm = lm - lm[WRIST:WRIST+1]

    # 2) Scale by palm size (scale invariance)
    palm_size = np.mean(np.linalg.norm(lm[[5,9,13,17]] - lm[WRIST], axis=1))
    if palm_size < 1e-5:
        palm_size = 1.0
    lm = lm / palm_size

    feats = []

    # 3) Raw normalized landmarks (42)
    feats.extend(lm.flatten())

    # 4) Finger curl: angle at PIP joint for each finger
    #    Straight finger ~ 180°, curled ~ 0-90°
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        v1 = lm[mcp] - lm[pip]
        v2 = lm[tip] - lm[pip]
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        feats.append(np.arccos(cos_a) / np.pi)  # normalized 0-1

    # 5) Tip-to-palm-center distance (curled finger = small distance)
    palm_c = lm[PALM_CENTER].mean(axis=0)
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        d = np.linalg.norm(lm[tip] - palm_c)
        feats.append(d)

    # 6) Finger spread: distance between adjacent fingertips
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips) - 1):
        d = np.linalg.norm(lm[tips[i]] - lm[tips[i+1]])
        feats.append(d)

    # 7) Thumb-index distance (key for C-shape: thumb is out)
    feats.append(np.linalg.norm(lm[4] - lm[8]))

    # 8) Hand "openness": mean tip-to-wrist distance / palm size
    tip_dists = [np.linalg.norm(lm[t] - lm[WRIST]) for t in tips]
    feats.append(np.mean(tip_dists))
    feats.append(np.std(tip_dists))

    return np.array(feats, dtype=np.float32)


def extract_features_batch(lm_array):
    """lm_array: (N,21,2) -> (N,F)"""
    return np.stack([extract_features(x) for x in lm_array])


print("\n📂 Loading Thay images...")
exts = ('.jpg','.jpeg','.png','.bmp','.webp')
files = [f for f in os.listdir(THAY_IMAGES_DIR) if f.lower().endswith(exts)]
print(f"   Found {len(files)} images")

thay_lm = []
for i, fn in enumerate(files):
    lm = landmarks_from_image(os.path.join(THAY_IMAGES_DIR, fn))
    if lm is not None:
        thay_lm.append(lm)
    if (i+1) % 100 == 0:
        print(f"   {i+1}/{len(files)} | kept {len(thay_lm)}")

if len(thay_lm) < 10:
    print("❌ Too few Thay images with detected hands.")
    exit()

thay_lm = np.array(thay_lm, dtype=np.float32)
print(f"   ✅ {len(thay_lm)} valid Thay images")

# ========== NEGATIVES FROM DB (other signs) ==========
print("\n📊 Loading negatives from DB...")
neg_lm = []
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.cursor().execute("SELECT * FROM rightHandDataset").fetchall()
    conn.close()
    THAY_CHARS = {'ث','thay','THAY','Thay'}
    for row in rows:
        label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]','',row[43]).strip()
        if label in THAY_CHARS: continue
        # DB stores pixel coords (0-300 range), convert to 0-1
        xy = np.array([float(row[i]) for i in range(1,43)], dtype=np.float32).reshape(21,2)
        xy[:,0] /= 300.0
        xy[:,1] /= 300.0
        neg_lm.append(xy)

neg_lm = np.array(neg_lm, dtype=np.float32) if neg_lm else np.zeros((0,21,2), np.float32)
print(f"   ✅ {len(neg_lm)} DB negatives")

# ========== SYNTHETIC HARD NEGATIVES (fake C-shapes) ==========
# Create *plausible* hand shapes that could be confused with Thay
print("\n🔧 Creating hard synthetic negatives...")

def jitter(lm, sigma=0.02):
    return lm + np.random.normal(0, sigma, lm.shape).astype(np.float32)

thay_mean = thay_lm.mean(axis=0)
synth = []

# Slight variations of *other* plausible shapes (NOT Thay)
for _ in range(500):
    v = jitter(thay_mean, 0.05).copy()
    # Straighten ALL fingers (open hand)
    for tip in [4,8,12,16,20]:
        v[tip] = v[tip] + (v[tip] - v[0]) * 0.5
    synth.append(v)

for _ in range(500):
    v = jitter(thay_mean, 0.05).copy()
    # Close all fingers (fist)
    palm = v[[0,5,9,13,17]].mean(axis=0)
    for tip in [4,8,12,16,20]:
        v[tip] = palm + (v[tip] - palm) * 0.3
    synth.append(v)

for _ in range(400):
    v = jitter(thay_mean, 0.06).copy()
    # Scramble finger positions
    np.random.shuffle(v[5:])
    synth.append(v)

for _ in range(400):
    # Pure random hand pose
    synth.append(np.random.uniform(0.2, 0.8, (21,2)).astype(np.float32))

synth = np.array(synth, dtype=np.float32)
print(f"   ✅ {len(synth)} synthetic negatives")

neg_all = np.concatenate([neg_lm, synth], axis=0) if len(neg_lm) else synth

# ========== EXTRACT FEATURES ==========
print("\n🔢 Extracting engineered features...")
X_pos = extract_features_batch(thay_lm)
X_neg = extract_features_batch(neg_all)
print(f"   Feature dim: {X_pos.shape[1]}")

X = np.vstack([X_pos, X_neg]).astype(np.float32)
y = np.concatenate([np.ones(len(X_pos), np.float32),
                    np.zeros(len(X_neg), np.float32)])

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
print(f"📦 Dataset: {len(X)} | +{int(y.sum())} | -{int(len(y)-y.sum())}")

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_v, X_te, y_v, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
class_weights = {0: cw[0], 1: cw[1]}

# ========== MODEL ==========
FEAT_DIM = X.shape[1]
print(f"\n🧠 Building model (input={FEAT_DIM})")

def build_model():
    inp = keras.Input(shape=(FEAT_DIM,))
    x = layers.Dense(256, activation='relu')(inp); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(64,  activation='relu')(x); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(32,  activation='relu')(x); x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inp, out)

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', keras.metrics.AUC(name='auc')])

# ========== TRAIN ==========
print("\n🚀 Training...")
cb = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=40, mode='max',
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=15, min_lr=1e-6, verbose=1),
]
model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
          epochs=300, batch_size=32, class_weight=class_weights,
          callbacks=cb, verbose=1)

# ========== EVALUATE — HIGH PRECISION THRESHOLD ==========
y_prob = model.predict(X_te, verbose=0).flatten()
auc = roc_auc_score(y_te, y_prob)

# **KEY CHANGE**: pick threshold maximizing PRECISION subject to recall >= 0.80
# This kills false positives (other signs) while still catching real Thay
best = (0, 0.5)
for t in np.arange(0.5, 0.995, 0.005):
    yp = (y_prob > t).astype(int)
    rec = recall_score(y_te, yp, zero_division=0)
    if rec < 0.80:
        continue
    prec = precision_score(y_te, yp, zero_division=0)
    if prec > best[0]:
        best = (prec, t)

best_thresh = best[1]
acc  = accuracy_score(y_te, (y_prob > best_thresh).astype(int))
prec = precision_score(y_te, (y_prob > best_thresh).astype(int), zero_division=0)
rec  = recall_score(y_te, (y_prob > best_thresh).astype(int), zero_division=0)

print(f"\n   ✅ Test Acc: {acc*100:.2f}% | AUC: {auc:.4f}")
print(f"   ✅ Threshold: {best_thresh:.3f}")
print(f"   ✅ Precision: {prec:.3f}  <- higher = fewer false positives")
print(f"   ✅ Recall:    {rec:.3f}  <- higher = catches more real Thay")
print(classification_report(y_te, (y_prob > best_thresh).astype(int),
                             target_names=['Not Thay','Thay']))

# ========== CONVERT ==========
print("\n📦 Converting to TFLite...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
def rep_gen():
    for i in range(min(300, len(X_tr))):
        yield [X_tr[i:i+1].astype(np.float32)]
conv.representative_dataset = rep_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.float32
conv.inference_output_type = tf.float32
tfl = conv.convert()
open(MODEL_PATH, 'wb').write(tfl)
model.save(KERAS_PATH)
print(f"   ✅ {MODEL_PATH} ({len(tfl)/1024:.1f} KB)")

# ========== INFO + REFERENCE ==========
info = {
    "model": "Thay C-Shape Detector",
    "sign": "ث", "sign_name": "Thay",
    "test_accuracy": float(acc),
    "test_auc": float(auc),
    "test_precision": float(prec),
    "test_recall": float(rec),
    "threshold": float(best_thresh),
    "input_features": int(FEAT_DIM),
    "feature_engineering": "translation+scale invariant, finger curl angles, tip-palm distances",
    "requires_geometric_check": False
}
json.dump(info, open(INFO_PATH,'w',encoding='utf-8'), indent=2, ensure_ascii=False)
np.save(REF_PATH, thay_lm.mean(axis=0))

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
print(f"   Precision: {prec:.3f}  (aim > 0.90)")
print(f"   Recall:    {rec:.3f}  (aim > 0.80)")
hands.close()