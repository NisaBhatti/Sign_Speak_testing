# Save as: training/train_hay_v4.py

import os, cv2, json, sqlite3, re
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

print("="*60)
print("SIGNSPEAK - Hay v4 (with FINGER-STATE features)")
print("="*60)

# ========== PATHS ==========
HAY_FOLDER = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Hay"
DB_PATH    = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
OUT_DIR    = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "hay_robust.tflite")
INFO_PATH  = os.path.join(OUT_DIR, "hay_robust_info.json")

# ========== FEATURE EXTRACTION ==========
def normalize_landmarks(pts):
    pts = np.array(pts, dtype=np.float32).reshape(21, 2)
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    return pts / scale

def finger_states_from_norm(norm_pts):
    """
    norm_pts: (21, 2) already normalized (wrist at origin).
    Returns [index, middle, ring, pinky, thumb] 1=open, 0=closed.
    """
    wrist = norm_pts[0]
    # ---- 4 fingers (index, middle, ring, pinky) ----
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    states = []
    for tip, pip in zip(tips, pips):
        d_tip = np.linalg.norm(norm_pts[tip] - wrist)
        d_pip = np.linalg.norm(norm_pts[pip] - wrist)
        states.append(1.0 if d_tip > d_pip else 0.0)

    # ---- thumb (compare tip to MCP horizontally) ----
    thumb_open = 1.0 if abs(norm_pts[4][0] - norm_pts[2][0]) > 0.15 else 0.0
    states.append(thumb_open)
    return states

def make_features(pts):
    """42 landmark + 5 finger-state = 47 features."""
    norm = normalize_landmarks(pts)
    fs = finger_states_from_norm(norm)
    return np.concatenate([norm.flatten(), fs]).astype(np.float32)

FEATURE_DIM = 47   # 42 + 5

# ========== 1. HAY POSITIVES ==========
print(f"\n📁 Loading Hay images from: {HAY_FOLDER}")
hands = mp.solutions.hands.Hands(static_image_mode=True,
                                 max_num_hands=1,
                                 min_detection_confidence=0.4)
files = [f for f in os.listdir(HAY_FOLDER)
         if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]

positives = []
for f in files:
    img = cv2.imread(os.path.join(HAY_FOLDER, f))
    if img is None: continue
    r = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if r.multi_hand_landmarks:
        pts = [[p.x, p.y] for p in r.multi_hand_landmarks[0].landmark]
        positives.append(pts)

print(f"   ✅ {len(positives)} Hay samples")
if len(positives) < 20:
    print("❌ Not enough Hay images."); exit()

# ========== 2. HARD NEGATIVES from DB ==========
print(f"\n📊 Loading non-Hay signs from DB...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT * FROM rightHandDataset")
rows = cur.fetchall()
conn.close()

negatives_db = []
for row in rows:
    label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]',
                   '', str(row[43])).strip()
    if label in ('هـ', 'ه', 'hay', 'Hay'):
        continue
    try:
        feats = [float(row[i]) for i in range(1, 43)]
    except Exception:
        continue
    pts = np.array(feats, dtype=np.float32).reshape(21, 2)
    pts[:, 0] /= 300.0
    pts[:, 1] /= 300.0
    negatives_db.append(pts.tolist())

print(f"   ✅ {len(negatives_db)} DB negatives")

# ========== 3. SYNTHETIC HARD NEGATIVES (3+ fingers open) ==========
# Build negatives by taking Hay landmarks and OPENING extra fingers.
# This forces the model to learn: Hay = exactly 2 fingers open.
print("\n🔄 Generating synthetic hard negatives (3+ fingers open)...")

def force_open_fingers(pts, open_indices):
    """Moves specified fingertip landmarks outward from wrist."""
    p = np.array(pts, dtype=np.float32).reshape(21, 2).copy()
    wrist = p[0]
    tip_map = {6: 8, 10: 12, 14: 16, 18: 20}  # pip -> tip
    for pip in open_indices:
        tip = tip_map[pip]
        direction = p[pip] - wrist
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        p[tip] = p[pip] + direction * 0.10   # push tip outward
    return p

synth_neg = []
for pts in positives:
    # Open 3rd finger (ring)
    synth_neg.append(force_open_fingers(pts, [14]))
    # Open 4th (pinky)
    synth_neg.append(force_open_fingers(pts, [18]))
    # Open 3rd and 4th
    synth_neg.append(force_open_fingers(pts, [14, 18]))
    # Open all 4
    synth_neg.append(force_open_fingers(pts, [6, 10, 14, 18]))
    # Open index only (like Alif)
    synth_neg.append(force_open_fingers(pts, [6]))

print(f"   ✅ {len(synth_neg)} synthetic hard negatives")

# ========== 4. AUGMENTATION ==========
def augment(pts, n=1):
    out = []
    p0 = np.array(pts, dtype=np.float32).reshape(21, 2)
    for _ in range(n):
        p = p0.copy()
        if np.random.rand() < 0.8:
            a = np.random.uniform(-25, 25) * np.pi / 180
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            p = p @ R.T
        if np.random.rand() < 0.8:
            p = p * np.random.uniform(0.85, 1.15)
        if np.random.rand() < 0.8:
            p = p + np.random.uniform(-0.08, 0.08, size=2)
        if np.random.rand() < 0.8:
            p = p + np.random.normal(0, 0.008, size=p.shape)
        if np.random.rand() < 0.3:
            p[:, 0] = -p[:, 0]
        out.append(p.tolist())
    return out

print("\n🔄 Normalizing + augmenting...")

# Build features
X_pos = [make_features(p) for p in positives]
for p in positives:
    for a in augment(p, 12):
        X_pos.append(make_features(a))
X_pos = np.array(X_pos, dtype=np.float32)

X_neg = [make_features(p) for p in negatives_db]
for p in negatives_db:
    for a in augment(p, 4):
        X_neg.append(make_features(a))
# Add synthetic hard negatives (also augmented)
for p in synth_neg:
    X_neg.append(make_features(p))
    for a in augment(p, 5):
        X_neg.append(make_features(a))
X_neg = np.array(X_neg, dtype=np.float32)

y_pos = np.ones(len(X_pos), dtype=np.float32)
y_neg = np.zeros(len(X_neg), dtype=np.float32)

print(f"   Positives: {len(X_pos)}")
print(f"   Negatives: {len(X_neg)}")

# Balance
n = min(len(X_pos), len(X_neg))
X_pos, y_pos = X_pos[:n], y_pos[:n]
X_neg, y_neg = X_neg[:n], y_neg[:n]

X = np.vstack([X_pos, X_neg])
y = np.hstack([y_pos, y_neg])
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"   Total: {len(X)} (balanced)")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

# ========== 5. MODEL ==========
print("\n🧠 Building model...")
model = keras.Sequential([
    layers.Input(shape=(FEATURE_DIM,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
model.summary()

# ========== 6. TRAIN ==========
# Bias slightly toward "Not Hay" so false positives drop
class_weight = {0: 1.3, 1: 1.0}

history = model.fit(
    X_tr, y_tr,
    validation_split=0.15,
    epochs=150,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=8, min_lr=1e-6),
    ],
    verbose=1
)

# ========== 7. EVALUATE + THRESHOLD ==========
print("\n📈 Evaluating...")
y_prob = model.predict(X_te, verbose=0).flatten()

acc = accuracy_score(y_te, (y_prob > 0.5).astype(int))
auc = roc_auc_score(y_te, y_prob)

best_t, best_f1, best_prec = 0.5, 0, 0
for t in np.arange(0.3, 0.95, 0.02):
    p = (y_prob > t).astype(int)
    tp = ((p==1)&(y_te==1)).sum(); fp = ((p==1)&(y_te==0)).sum()
    fn = ((p==0)&(y_te==1)).sum()
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    if prec >= 0.92 and f1 > best_f1:
        best_f1, best_t, best_prec = f1, t, prec

if best_f1 == 0:
    best_t = 0.65

print(f"   Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}")
print(f"   Threshold: {best_t:.2f}  (precision={best_prec:.3f}, F1={best_f1:.3f})")
print(confusion_matrix(y_te, (y_prob > best_t).astype(int)))
print(classification_report(y_te, (y_prob > best_t).astype(int),
                            target_names=['Not-Hay', 'Hay']))

# ========== 8. SAVE ==========
print("\n💾 Saving TFLite...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
with open(MODEL_PATH, 'wb') as f:
    f.write(conv.convert())

info = {
    "model": "hay_robust",
    "version": 4,
    "sign": "هـ",
    "sign_name": "Hay",
    "features": "42_landmarks + 5_finger_states",
    "feature_dim": FEATURE_DIM,
    "normalization": "wrist_centered_scale_invariant",
    "test_accuracy": float(acc),
    "test_auc": float(auc),
    "threshold": float(best_t),
    "num_hay_samples": int(len(positives)),
    "num_negative_samples": int(len(X_neg)),
    "input_shape": [FEATURE_DIM],
    "description": "Hay = 2 fingers open (index+middle), others closed"
}
with open(INFO_PATH, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"✅ Model: {MODEL_PATH}")
print(f"✅ Info:  {INFO_PATH}")
print(f"\n🎯 Acc: {acc*100:.1f}%  |  AUC: {auc:.4f}  |  Thr: {best_t:.2f}")