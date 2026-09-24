# Save as: test_tay.py
# Live webcam Tay (ت) detection — uses SAME normalization as training.

import cv2, os, time
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ============ CONFIG ============
MODEL_DIR   = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_tay"
TFLITE_PATH = os.path.join(MODEL_DIR, "tay_robust.tflite")
H5_PATH     = os.path.join(MODEL_DIR, "tay_robust.h5")
THRESHOLD   = 0.5      # 0.6-0.7 = stricter, 0.4 = more sensitive
SMOOTH_N    = 7        # temporal majority vote window
VOTE_MIN    = 5        # need ≥VOTE_MIN of last SMOOTH_N frames above threshold

WRIST, MID_MCP = 0, 9

print("="*60)
print("SIGNSPEAK - LIVE TAY (ت) DETECTION")
print("="*60)

# ============ LOAD ============
if os.path.exists(TFLITE_PATH):
    print(f"📦 TFLite: {TFLITE_PATH}")
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()
    def predict(features):
        interp.set_tensor(in_det[0]['index'], np.array([features], dtype=np.float32))
        interp.invoke()
        return float(interp.get_tensor(out_det[0]['index'])[0][0])
elif os.path.exists(H5_PATH):
    print(f"📦 Keras: {H5_PATH}")
    model = tf.keras.models.load_model(H5_PATH)
    def predict(features):
        return float(model.predict(np.array([features], dtype=np.float32), verbose=0)[0][0])
else:
    raise SystemExit("❌ No model found. Run train_tay.py first.")

# ============ SAME NORMALIZATION AS TRAINING ============
def normalize_landmarks(landmarks_xy, handedness):
    pts = np.asarray(landmarks_xy, dtype=np.float32).reshape(21, 2)
    pts = pts - pts[WRIST]
    if handedness == 'Right':
        pts[:, 0] = -pts[:, 0]
    scale = np.linalg.norm(pts[MID_MCP])
    if scale < 1e-6:
        scale = 1e-6
    pts = pts / scale
    return pts.flatten().astype(np.float32)

# ============ MEDIAPIPE ============
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.4, min_tracking_confidence=0.4)

# ============ CAMERA ============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ Cannot open webcam")

print("\n🎥 Live. Press 'q' or ESC to quit.\n")

history = []          # rolling window of (prob > threshold) booleans
last_prob = 0.0
last_text = "No Hand"
fps, last_t = 0.0, time.time()

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)         # mirror for user
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    status_text  = "No Hand"
    status_color = (128, 128, 128)
    last_prob    = 0.0

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # MediaPipe handedness is from IMAGE POV. Because we flipped
        # the frame, the label is reversed vs the user's real hand —
        # this is fine as long as training and testing use the SAME rule,
        # which they do (both read MediaPipe output directly).
        hd = res.multi_handedness[0].classification[0].label  # 'Left' / 'Right'

        pts = [(p.x, p.y) for p in lm.landmark]
        feats = normalize_landmarks(pts, hd)

        prob = predict(feats)
        last_prob = prob

        history.append(prob > THRESHOLD)
        if len(history) > SMOOTH_N:
            history.pop(0)

        votes = sum(history)
        if votes >= VOTE_MIN:
            status_text  = f"TAY (ت)   {prob*100:.1f}%"
            status_color = (0, 220, 0)
        else:
            status_text  = f"Not Tay   {prob*100:.1f}%"
            status_color = (0, 0, 255)
    else:
        history.clear()

    last_text = status_text

    # HUD
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(frame, status_text, (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

    bar = int(w * min(max(last_prob, 0), 1))
    cv2.rectangle(frame, (0, h - 20), (bar, h), (0, 220, 0), -1)
    cv2.rectangle(frame, (0, h - 20), (w, h), (255, 255, 255), 1)

    now = time.time()
    fps = 0.9*fps + 0.1*(1.0/max(now-last_t, 1e-6))
    last_t = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("SignSpeak - Tay (ت) Detection", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n👋 Stopped. Last: {last_text}")