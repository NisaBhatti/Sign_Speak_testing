# Save as: D:\MODEL\Sign_Speak_testing-main\Testing\thay_test.py

import cv2, numpy as np, tensorflow as tf, mediapipe as mp, json, os
from collections import deque

print("="*60)
print("SIGNSPEAK - Thay C-Shape Detection")
print("="*60)

PROJECT_ROOT = r"D:\MODEL\Sign_Speak_testing-main"
MODEL_PATH = os.path.join(PROJECT_ROOT, "Exported_Model", "thay_robust.tflite")
INFO_PATH  = os.path.join(PROJECT_ROOT, "Exported_Model", "thay_robust_info.json")
REF_PATH   = os.path.join(PROJECT_ROOT, "Exported_Model", "thay_reference.npy")

# ========== LOAD ==========
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()
out_det = interpreter.get_output_details()
info = json.load(open(INFO_PATH, 'r', encoding='utf-8'))
THRESHOLD = info['threshold']
print(f"   Model: {info['model']}")
print(f"   Acc: {info['test_accuracy']*100:.1f}% | AUC: {info['test_auc']:.4f}")
print(f"   Precision: {info['test_precision']:.3f} | Recall: {info['test_recall']:.3f}")
print(f"   Threshold: {THRESHOLD}")

ref_thay = np.load(REF_PATH) if os.path.exists(REF_PATH) else None

# ========== SAME FEATURE ENGINEERING AS TRAINING ==========
FINGER_JOINTS = {
    'thumb':  [1,2,3,4], 'index': [5,6,7,8], 'middle': [9,10,11,12],
    'ring':   [13,14,15,16], 'pinky': [17,18,19,20],
}
WRIST = 0
PALM_CENTER = [0,5,9,13,17]

def extract_features(lm_xy):
    lm = lm_xy.astype(np.float32)
    lm = lm - lm[WRIST:WRIST+1]
    palm_size = np.mean(np.linalg.norm(lm[[5,9,13,17]] - lm[WRIST], axis=1))
    if palm_size < 1e-5: palm_size = 1.0
    lm = lm / palm_size

    feats = list(lm.flatten())

    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        v1 = lm[mcp] - lm[pip]; v2 = lm[tip] - lm[pip]
        n1 = np.linalg.norm(v1) + 1e-6; n2 = np.linalg.norm(v2) + 1e-6
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        feats.append(np.arccos(cos_a) / np.pi)

    palm_c = lm[PALM_CENTER].mean(axis=0)
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        feats.append(np.linalg.norm(lm[tip] - palm_c))

    tips = [4,8,12,16,20]
    for i in range(len(tips)-1):
        feats.append(np.linalg.norm(lm[tips[i]] - lm[tips[i+1]]))
    feats.append(np.linalg.norm(lm[4] - lm[8]))

    tip_dists = [np.linalg.norm(lm[t] - lm[WRIST]) for t in tips]
    feats.append(np.mean(tip_dists))
    feats.append(np.std(tip_dists))

    return np.array(feats, dtype=np.float32)

# ========== MEDIAPIPE ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
WIN = 'Thay C-Shape Detection'
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1100, 800)

print("\n" + "="*60)
print("🎯 THAY (ث) C-SHAPE DETECTION")
print("="*60)
print("   Show the C-shape sign. Other signs will be rejected.")
print("   Q=Quit  S=Save")
print("="*60 + "\n")

pred_hist = deque(maxlen=10)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_xy = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)

            # Draw reference
            if ref_thay is not None:
                for rx, ry in ref_thay:
                    cv2.circle(frame, (int(rx*w), int(ry*h)), 4, (255,255,255), -1)
                for c in mp_hands.HAND_CONNECTIONS:
                    cv2.line(frame,
                             (int(ref_thay[c[0]][0]*w), int(ref_thay[c[0]][1]*h)),
                             (int(ref_thay[c[1]][0]*w), int(ref_thay[c[1]][1]*h)),
                             (255,255,255), 1)

            # Draw user hand
            for x, y in lm_xy:
                cv2.circle(frame, (int(x*w), int(y*h)), 6, (0,255,0), -1)
            for c in mp_hands.HAND_CONNECTIONS:
                cv2.line(frame,
                         (int(lm_xy[c[0]][0]*w), int(lm_xy[c[0]][1]*h)),
                         (int(lm_xy[c[1]][0]*w), int(lm_xy[c[1]][1]*h)),
                         (0,255,0), 2)

            # Predict
            feats = extract_features(lm_xy).reshape(1, -1).astype(np.float32)
            interpreter.set_tensor(in_det[0]['index'], feats)
            interpreter.invoke()
            raw = float(interpreter.get_tensor(out_det[0]['index'])[0][0])

            pred_hist.append(raw)
            smooth = sum(pred_hist) / len(pred_hist)
            is_thay = smooth > THRESHOLD

            # UI
            cv2.rectangle(frame, (0,0), (w, 95), (0,0,0), -1)
            if is_thay:
                if smooth > min(0.95, THRESHOLD + 0.15):
                    text = "✅ THAY (ث) - CONFIRMED"
                    color = (0, 255, 0)
                else:
                    text = "✅ THAY (ث) - DETECTED"
                    color = (0, 255, 128)
            else:
                if smooth > THRESHOLD - 0.15:
                    text = "⚠️ UNCERTAIN - Adjust hand"
                    color = (0, 165, 255)
                else:
                    text = "❌ NOT THAY"
                    color = (0, 0, 255)

            cv2.putText(frame, text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Score: {smooth*100:.1f}%  (threshold {THRESHOLD*100:.0f}%)",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"Raw: {raw:.3f} | Smooth: {smooth:.3f}",
                        (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    else:
        pred_hist.clear()
        cv2.rectangle(frame, (0,0), (w,70), (0,0,0), -1)
        cv2.putText(frame, "Show your hand", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
        if ref_thay is not None:
            for rx, ry in ref_thay:
                cv2.circle(frame, (int(rx*w), int(ry*h)), 4, (100,100,100), -1)

    cv2.imshow(WIN, frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('s'):
        cv2.imwrite(f"thay_{frame_count}.png", frame)

cap.release(); cv2.destroyAllWindows(); hands.close()
print(f"\n✅ Done. {frame_count} frames.")