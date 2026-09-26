# ============================================================
# Save as: train_all_letters.py
# Trains ONE model on all Arabic letter folders.
# Saves BOTH the Arabic letter and English name for each class.
# ============================================================

import os, glob, json, random
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from letters_map import FOLDER_TO_ARABIC

print("=" * 60)
print("SIGNSPEAK - Accurate Multi-Letter Trainer")
print("=" * 60)

# ========== CONFIG ==========
DATASET_ROOT = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset"
OUTPUT_DIR   = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED  = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
MIN_IMAGES_PER_CLASS = 8

# ============================================================
# 1) SCAN FOLDERS — merge aliases by Arabic letter
# ============================================================
print(f"\n🔎 Scanning: {DATASET_ROOT}")

letter_to_imgs  = {}
letter_to_names = {}

for name in sorted(os.listdir(DATASET_ROOT)):
    path = os.path.join(DATASET_ROOT, name)
    if not os.path.isdir(path): continue
    if name not in FOLDER_TO_ARABIC:
        print(f"   ⚠️ Skipping unknown folder: {name}")
        continue
    ar, en = FOLDER_TO_ARABIC[name]

    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        imgs += glob.glob(os.path.join(path, ext))
        imgs += glob.glob(os.path.join(path, "**", ext), recursive=True)
    imgs = list(set(imgs))

    letter_to_imgs.setdefault(ar, []).extend(imgs)
    letter_to_names.setdefault(ar, []).append(en)

print("\n📋 Letters discovered:")
for ar in sorted(letter_to_imgs):
    names = letter_to_names[ar]
    n = len(letter_to_imgs[ar])
    flag = "✅" if n >= MIN_IMAGES_PER_CLASS else "⚠️"
    print(f"   {flag} {ar}  ({', '.join(names)}): {n} images")

kept_letters = sorted([ar for ar, imgs in letter_to_imgs.items()
                       if len(imgs) >= MIN_IMAGES_PER_CLASS])
if len(kept_letters) < 2:
    raise RuntimeError("❌ Need at least 2 letters with enough images.")

print(f"\n   Keeping {len(kept_letters)} classes: {kept_letters}")

# Build final English-name list in the same order
english_names = []
for ar in kept_letters:
    english_names.append(letter_to_names[ar][0])

# ============================================================
# 2) EXTRACT LANDMARKS
# ============================================================
print(f"\n🖼️ Extracting landmarks...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

X_all, y_all = [], []
per_class_kept = {}

for cls_idx, ar in enumerate(kept_letters):
    imgs = letter_to_imgs[ar]
    kept = 0
    for path in imgs:
        img = cv2.imread(path)
        if img is None: continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks: continue
        lm = res.multi_hand_landmarks[0]
        feats = []
        for p in lm.landmark:
            feats.append(p.x)
            feats.append(p.y)
        if len(feats) != 42: continue
        X_all.append(feats)
        y_all.append(cls_idx)
        kept += 1
    per_class_kept[ar] = kept
    print(f"   {ar} ({english_names[cls_idx]}): {kept} landmarks  "
          f"(from {len(imgs)} images)")

hands.close()

valid_letters = [ar for ar in kept_letters if per_class_kept[ar] > 0]
if len(valid_letters) < 2:
    raise RuntimeError("❌ Too few classes with usable landmarks.")

old_to_new = {kept_letters.index(ar): i for i, ar in enumerate(valid_letters)}
X_all = np.array([x for x, y in zip(X_all, y_all) if y in old_to_new], dtype=np.float32)
y_all = np.array([old_to_new[y] for y in y_all if y in old_to_new], dtype=np.int64)
arabic_letters = valid_letters
english_names  = [english_names[kept_letters.index(ar)] for ar in valid_letters]

print(f"\n   Total samples: {len(X_all)}")
print(f"   Classes: {list(zip(english_names, arabic_letters))}")

X_all = np.clip(X_all, 0.0, 1.0)

# ============================================================
# 3) AUGMENT
# ============================================================
print("\n🔧 Augmenting...")
rng = np.random.default_rng(RANDOM_SEED)

def augment(X, y):
    augX, augY = [X], [y]
    for _ in range(2):
        n = rng.normal(0, 0.015, X.shape).astype(np.float32)
        augX.append(np.clip(X + n, 0, 1)); augY.append(y)
    for _ in range(2):
        s = rng.uniform(0.92, 1.08, len(X))[:, None]
        augX.append(np.clip(X * s, 0, 1)); augY.append(y)
    for _ in range(2):
        dx = rng.uniform(-0.03, 0.03, len(X))[:, None]
        dy = rng.uniform(-0.03, 0.03, len(X))[:, None]
        Xs = X.copy()
        Xs[:, 0::2] = np.clip(Xs[:, 0::2] + dx, 0, 1)
        Xs[:, 1::2] = np.clip(Xs[:, 1::2] + dy, 0, 1)
        augX.append(Xs); augY.append(y)
    # horizontal flip
    Xf = X.copy()
    Xf[:, 0::2] = 1.0 - Xf[:, 0::2]
    augX.append(Xf); augY.append(y)
    return np.vstack(augX), np.hstack(augY)

X_aug, y_aug = augment(X_all, y_all)
print(f"   {len(X_all)} → {len(X_aug)}")

sh = rng.permutation(len(X_aug))
X_aug, y_aug = X_aug[sh], y_aug[sh]

# ============================================================
# 4) SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
print(f"\n✂️ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {i: float(w) for i, w in enumerate(cw)}
print(f"⚖️ Class weights: {cw_dict}")

# ============================================================
# 5) MODEL
# ============================================================
num_classes = len(arabic_letters)
print(f"\n🏗️ Building model ({num_classes} classes)...")

model = Sequential([
    Dense(512, activation='relu', input_shape=(42,)),
    BatchNormalization(), Dropout(0.4),
    Dense(512, activation='relu'), BatchNormalization(), Dropout(0.4),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(256, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'), BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'), BatchNormalization(), Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ============================================================
# 6) TRAIN
# ============================================================
print("\n🚀 Training...")
cbs = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'all_letters.h5'),
                    monitor='val_accuracy', save_best_only=True, verbose=1)
]
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300, batch_size=64,
    callbacks=cbs, class_weight=cw_dict, verbose=1
)

# ============================================================
# 7) PER-CLASS ACCURACY
# ============================================================
print("\n📋 Per-letter accuracy (test set):")
test_preds = model.predict(X_test, verbose=0).argmax(axis=1)
per_class_acc = {}
for i, (ar, en) in enumerate(zip(arabic_letters, english_names)):
    mask = y_test == i
    if mask.sum() == 0:
        per_class_acc[ar] = None
        print(f"   {en:8s} ({ar}): NO TEST SAMPLES")
        continue
    acc = (test_preds[mask] == i).mean() * 100
    per_class_acc[ar] = acc
    bar = "█" * int(acc // 5) + "░" * (20 - int(acc // 5))
    print(f"   {en:8s} ({ar}): {acc:6.2f}%  {bar}   n_orig={per_class_kept[ar]}")

print("\n🔀 Top confusion pairs:")
confusions = {}
for t, p in zip(y_test, test_preds):
    if t == p: continue
    key = (english_names[t], english_names[p])
    confusions[key] = confusions.get(key, 0) + 1
for (t, p), c in sorted(confusions.items(), key=lambda kv: -kv[1])[:10]:
    print(f"   {c:4d}×  actual {t:8s} predicted as {p}")

# ============================================================
# 8) SAVE
# ============================================================
print("\n💾 Saving...")
model.save(os.path.join(OUTPUT_DIR, 'all_letters.h5'))

conv = tf.lite.TFLiteConverter.from_keras_model(model)
tfl = conv.convert()
with open(os.path.join(OUTPUT_DIR, 'all_letters.tflite'), 'wb') as f:
    f.write(tfl)
print(f"   TFLite: {len(tfl)/1024:.1f} KB")

_, te_acc = model.evaluate(X_test, y_test, verbose=0)
info = {
    'num_classes': num_classes,
    'arabic_letters': arabic_letters,   # ['ا','ب',...]
    'english_names':  english_names,    # ['Alif','Bey',...]
    'input': '[x0,y0,x1,y1,...,x20,y20] MediaPipe landmarks in 0-1 range',
    'normalization': 'NONE — MediaPipe already returns 0-1',
    'test_accuracy': float(te_acc),
    'per_class_accuracy': {k: (None if v is None else float(v))
                            for k, v in per_class_acc.items()},
}
with open(os.path.join(OUTPUT_DIR, 'all_letters_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\n✅ DONE — Test Acc={te_acc*100:.2f}%")