# Save as: prepare_ray_data.py
import sqlite3, cv2, mediapipe as mp, os, glob, sys

DB_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\main_dataset.db"
RAY_IMAGE_FOLDER = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Ray"
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
RAY_LETTER = 'ر'
TABLE_NAME = "rightHandDataset"

print("=" * 60)
print("PREPARE RAY DATA")
print("=" * 60)

if not os.path.exists(DB_PATH):
    sys.exit(f"❌ DB not found: {DB_PATH}")
if not os.path.exists(RAY_IMAGE_FOLDER):
    sys.exit(f"❌ Folder not found: {RAY_IMAGE_FOLDER}")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
cols = cursor.fetchall()
col_names = [c[1] for c in cols]
n_cols = len(col_names)

label_col_idx = n_cols - 1
for i, name in enumerate(col_names):
    if name.lower() in ('label', 'letter', 'rletter', 'class', 'sign'):
        label_col_idx = i
        break
feature_start = 1 if col_names[0].lower() in ('id', 'rowid') else 0

cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE TRIM({col_names[label_col_idx]}) = ?", (RAY_LETTER,))
existing = cursor.fetchone()[0]
print(f"Existing Ray samples: {existing}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.3, min_tracking_confidence=0.3)

files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
    files.extend(glob.glob(os.path.join(RAY_IMAGE_FOLDER, ext)))
print(f"Found {len(files)} images")

added = skipped = failed = 0
for p in files:
    img = cv2.imread(p)
    if img is None:
        failed += 1
        continue
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        skipped += 1
        continue

    feats = []
    for lm in res.multi_hand_landmarks[0].landmark:
        feats.append(lm.x * IMAGE_WIDTH)
        feats.append(lm.y * IMAGE_HEIGHT)

    row = [None] * n_cols
    for i, v in enumerate(feats):
        row[feature_start + i] = float(v)
    row[label_col_idx] = RAY_LETTER
    for i in range(n_cols):
        if row[i] is None and i != 0 and i != label_col_idx:
            row[i] = 0.0

    try:
        cursor.execute(f"INSERT INTO {TABLE_NAME} VALUES ({','.join(['?']*n_cols)})", row)
        added += 1
    except Exception as e:
        failed += 1
        if failed <= 3:
            print(f"  ⚠️ {os.path.basename(p)}: {e}")

conn.commit()
cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE TRIM({col_names[label_col_idx]}) = ?", (RAY_LETTER,))
new_count = cursor.fetchone()[0]
conn.close()

print(f"\n✅ Added: {added}  ⏭️ Skipped: {skipped}  ❌ Failed: {failed}")
print(f"📈 Ray total: {existing} → {new_count}")
if added > 0:
    print("\nNext: python train_ray.py")