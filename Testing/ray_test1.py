# Save as: test_ray.py
import sqlite3, numpy as np, tensorflow as tf, cv2, mediapipe as mp
import re, os, glob, json
from collections import Counter

# ======================= CONFIG =======================
DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
RAY_IMAGE_FOLDER = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Ray"
MODEL_DIR = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Exported_Model\exported_models_ray"
TABLE_NAME = "rightHandDataset"
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
RAY_LETTER = 'ر'
# ======================================================

KERAS_PATH = os.path.join(MODEL_DIR, 'ray_robust.h5')
TFLITE_PATH = os.path.join(MODEL_DIR, 'ray_robust.tflite')
INFO_PATH = os.path.join(MODEL_DIR, 'ray_robust_info.json')

print("=" * 60)
print("TEST RAY (ر) — STRICT")
print("=" * 60)

# ---------- Load threshold ----------
THRESHOLD = 0.5
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, encoding='utf-8') as f:
        info = json.load(f)
    THRESHOLD = float(info.get('threshold', 0.5))
    print(f"Threshold from info.json: {THRESHOLD:.3f}")
    print(f"Training avg Ray confidence: {info.get('avg_ray_confidence', 0)*100:.1f}%")

# ---------- Load model ----------
USE_TFLITE = False
interpreter = None
if os.path.exists(TFLITE_PATH):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    USE_TFLITE = True
    print(f"✅ TFLite loaded")

model = None
if not USE_TFLITE:
    model = tf.keras.models.load_model(KERAS_PATH)
    print(f"✅ Keras loaded")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.3, min_tracking_confidence=0.3)

def lm_from_img(path):
    img = cv2.imread(path)
    if img is None:
        return None
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    out = []
    for lm in res.multi_hand_landmarks[0].landmark:
        out.append(lm.x * IMAGE_WIDTH)
        out.append(lm.y * IMAGE_HEIGHT)
    return np.array(out, dtype=np.float32)

def lm_from_frame(frame):
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None, None
    out = []
    for lm in res.multi_hand_landmarks[0].landmark:
        out.append(lm.x * IMAGE_WIDTH)
        out.append(lm.y * IMAGE_HEIGHT)
    return np.array(out, dtype=np.float32), res.multi_hand_landmarks[0]

def predict(lm):
    if lm is None:
        return 0.0
    x = lm.copy()
    x[0::2] /= IMAGE_WIDTH
    x[1::2] /= IMAGE_HEIGHT
    x = x.reshape(1, -1).astype(np.float32)
    if USE_TFLITE:
        interpreter.set_tensor(in_det[0]['index'], x)
        interpreter.invoke()
        return float(interpreter.get_tensor(out_det[0]['index'])[0][0])
    return float(model.predict(x, verbose=0)[0][0])

def predict_batch(X):
    if X is None or len(X) == 0:
        return np.array([])
    Xn = X.copy()
    Xn[:, 0::2] /= IMAGE_WIDTH
    Xn[:, 1::2] /= IMAGE_HEIGHT
    if USE_TFLITE:
        out = []
        for r in Xn:
            interpreter.set_tensor(in_det[0]['index'], r.reshape(1, -1))
            interpreter.invoke()
            out.append(float(interpreter.get_tensor(out_det[0]['index'])[0][0]))
        return np.array(out)
    return model.predict(Xn, verbose=0).flatten()

# ---------- TEST 1: Ray folder ----------
print(f"\n{'='*60}\nTEST 1 — Ray folder\n{'='*60}")
if os.path.exists(RAY_IMAGE_FOLDER):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        files.extend(glob.glob(os.path.join(RAY_IMAGE_FOLDER, ext)))
    print(f"{len(files)} images")

    det = no_hand = 0
    confs = []
    for p in files:
        lm = lm_from_img(p)
        if lm is None:
            no_hand += 1
            continue
        c = predict(lm)
        confs.append(c)
        if c > THRESHOLD:
            det += 1

    valid = len(files) - no_hand
    if valid:
        print(f"✅ Detected: {det}/{valid} ({det/valid*100:.1f}%)")
        print(f"📊 Avg: {np.mean(confs):.4f}  Min: {np.min(confs):.4f}  Max: {np.max(confs):.4f}")
    print(f"⚠️ No hand: {no_hand}")

# ---------- TEST 2: DB — per class FP ----------
print(f"\n{'='*60}\nTEST 2 — Per-class false positives\n{'='*60}")

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

by_class = {}
for row in rows:
    try:
        feats = [float(row[i]) for i in feature_indices]
        label = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2066-\u2069]', '', str(row[label_idx])).strip()
        by_class.setdefault(label, []).append(feats)
    except (ValueError, IndexError):
        continue

print(f"\n{'Letter':>6s}  {'Count':>5s}  {'Detected':>9s}  {'FP rate':>8s}  Status")
print("-" * 55)

total_fp = 0
total_other = 0
for lbl in sorted(by_class.keys()):
    X_cls = np.array(by_class[lbl], dtype=np.float32)
    preds = predict_batch(X_cls)
    detected = (preds > THRESHOLD).sum()

    if lbl == RAY_LETTER:
        status = "✅ RAY (should detect)"
    else:
        total_fp += detected
        total_other += len(X_cls)
        rate = detected / len(X_cls)
        if rate == 0:
            status = "✅ clean"
        elif rate < 0.05:
            status = "✅ <5%"
        elif rate < 0.15:
            status = "⚠️ 5-15%"
        else:
            status = "❌ HIGH"

    print(f"{lbl:>6s}  {len(X_cls):>5d}  {detected:>9d}  {detected/len(X_cls)*100:>7.1f}%  {status}")

print("-" * 55)
print(f"\nTotal false positives on OTHER signs: {total_fp}/{total_other} "
      f"({total_fp/total_other*100:.2f}%)")

# ---------- TEST 3: Live camera ----------
print(f"\n{'='*60}\nTEST 3 — Live camera (q to quit)\n{'='*60}")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        lm, hand_lms = lm_from_frame(frame)
        c = predict(lm) if lm is not None else 0.0
        is_ray = c > THRESHOLD

        if hand_lms is not None:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        color = (0, 255, 0) if is_ray else (0, 0, 255)
        cv2.putText(frame, "RAY (ر)" if is_ray else "NOT RAY",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Conf: {c*100:.1f}%  Thr: {THRESHOLD:.2f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        bar = int(c * 300)
        cv2.rectangle(frame, (10, 95), (310, 120), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 95), (10 + bar, 120), color, -1)
        cv2.imshow("Ray (ر) Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

print(f"\n{'='*60}\n✅ DONE\n{'='*60}")
print("\nIf OTHER signs still get detected:")
print("  1. Run the training again — hard negatives get auto-weighted")
print("  2. Or manually raise threshold in ray_robust_info.json (e.g. 0.75)")
print("  3. Or add more images of the false-positive sign to DB")