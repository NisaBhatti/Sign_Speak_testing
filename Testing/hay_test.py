# Save as: Testing/hay_test.py

import os, cv2, json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

print("="*60)
print("SIGNSPEAK - Hay (هـ) Live Detection v4")
print("="*60)

# ========== PATHS ==========
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\hay_robust.tflite"
INFO_PATH  = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\hay_robust_info.json"
HAY_FOLDER = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Hay"

# ========== LOAD MODEL ==========
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_d = interpreter.get_input_details()
out_d = interpreter.get_output_details()

with open(INFO_PATH, 'r', encoding='utf-8') as f:
    info = json.load(f)

threshold  = float(info['threshold'])
FEATURE_DIM = int(info.get('feature_dim', 47))

print(f"   Model:     {info['model']} v{info.get('version',1)}")
print(f"   Features:  {info.get('features', 'raw landmarks')}  (dim={FEATURE_DIM})")
print(f"   Accuracy:  {info['test_accuracy']*100:.1f}%")
print(f"   AUC:       {info['test_auc']:.4f}")
print(f"   Threshold: {threshold:.3f}")

# ========== FEATURE ENGINEERING (must match training) ==========
def normalize_landmarks(pts):
    pts = np.array(pts, dtype=np.float32).reshape(21, 2)
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    return pts / scale

def finger_states_from_norm(norm_pts):
    wrist = norm_pts[0]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    states = []
    for tip, pip in zip(tips, pips):
        d_tip = np.linalg.norm(norm_pts[tip] - wrist)
        d_pip = np.linalg.norm(norm_pts[pip] - wrist)
        states.append(1.0 if d_tip > d_pip else 0.0)
    thumb_open = 1.0 if abs(norm_pts[4][0] - norm_pts[2][0]) > 0.15 else 0.0
    states.append(thumb_open)
    return states

def make_features(pts):
    norm = normalize_landmarks(pts)
    fs = finger_states_from_norm(norm)
    return np.concatenate([norm.flatten(), fs]).astype(np.float32)

# ========== REFERENCE SKELETON ==========
print("\n📊 Building reference Hay pose from images...")
mp_hands = mp.solutions.hands
static_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                              min_detection_confidence=0.4)

ref_norm = None
if os.path.exists(HAY_FOLDER):
    files = [f for f in os.listdir(HAY_FOLDER)
             if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    lms = []
    for f in files[:300]:
        img = cv2.imread(os.path.join(HAY_FOLDER, f))
        if img is None: continue
        r = static_hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if r.multi_hand_landmarks:
            pts = [[p.x, p.y] for p in r.multi_hand_landmarks[0].landmark]
            lms.append(normalize_landmarks(pts))
    if lms:
        ref_norm = np.mean(lms, axis=0)
        print(f"   ✅ Reference from {len(lms)} images")
static_hands.close()

if ref_norm is None:
    ref_norm = np.array([[0,0]]*21, dtype=np.float32)
    print("   ⚠️ Fallback reference")

def ref_to_screen(ref, w, h):
    pts = ref.copy()
    pts[:, 0] = pts[:, 0] * 0.35 + 0.5
    pts[:, 1] = pts[:, 1] * 0.35 + 0.55
    return pts

# ========== LIVE TRACKER ==========
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ========== CAMERA ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera"); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW = 'Hay Detection'
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1100, 800)

print("\n" + "="*60)
print("🎯 HAY (هـ) DETECTION — LIVE")
print("="*60)
print("   Target pose: 2 fingers straight (index+middle)")
print("                other fingers closed")
print("   Q = Quit    S = Save")
print("="*60 + "\n")

hist = deque(maxlen=10)
frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok: continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Reference skeleton (gray)
    ref_screen = ref_to_screen(ref_norm, w, h)
    for conn in mp_hands.HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(ref_screen[conn[0]][0]*w), int(ref_screen[conn[0]][1]*h)),
                 (int(ref_screen[conn[1]][0]*w), int(ref_screen[conn[1]][1]*h)),
                 (150,150,150), 1)
    for x, y in ref_screen:
        cv2.circle(frame, (int(x*w), int(y*h)), 4, (180,180,180), -1)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]

        # Draw user's hand
        for conn in mp_hands.HAND_CONNECTIONS:
            s = lm.landmark[conn[0]]; e = lm.landmark[conn[1]]
            cv2.line(frame, (int(s.x*w), int(s.y*h)),
                     (int(e.x*w), int(e.y*h)), (0,255,0), 2)
        for p in lm.landmark:
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 6, (0,255,0), -1)

        # ---- Predict ----
        raw_pts = [[p.x, p.y] for p in lm.landmark]
        feat = make_features(raw_pts).reshape(1, -1)
        interpreter.set_tensor(inp_d[0]['index'], feat)
        interpreter.invoke()
        pred = float(interpreter.get_tensor(out_d[0]['index'])[0][0])

        hist.append(pred)
        smooth = sum(hist)/len(hist)

        # Finger-state readout for on-screen feedback
        norm = normalize_landmarks(raw_pts)
        fs = finger_states_from_norm(norm)
        fingers_txt = "".join("🖐" if x else "✊" for x in fs[:4])

        is_hay = smooth > threshold
        color = (0,255,0) if is_hay else (0,0,255)
        label = "✅ HAY (هـ)" if is_hay else "❌ NOT HAY"

        # Top bar
        cv2.rectangle(frame, (0,0), (w,110), (0,0,0), -1)
        cv2.putText(frame, f"{label}  {smooth*100:.1f}%", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Fingers (Idx,Mid,Ring,Pinky): {int(fs[0])} {int(fs[1])} {int(fs[2])} {int(fs[3])}",
                    (20,72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, f"raw={pred:.3f}  smooth={smooth:.3f}  thr={threshold:.2f}",
                    (20,98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    else:
        hist.append(0.0)
        cv2.rectangle(frame, (0,0), (w,70), (0,0,0), -1)
        cv2.putText(frame, "Show your hand...", (20,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

    cv2.imshow(WINDOW, frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('s'):
        cv2.imwrite(f"hay_live_{frame_count}.png", frame)
        print(f"📸 saved hay_live_{frame_count}.png")

cap.release(); cv2.destroyAllWindows(); hands.close()
print(f"\n✅ Done. {frame_count} frames")