# ============================================================
# Save as: noon_test.py
# Only prints "NOON" if the hand pose matches the Noon sign
# for MANY consecutive frames with high confidence.
# ============================================================

import cv2, os, json
import numpy as np
import tensorflow as tf
import mediapipe as mp

print("=" * 60)
print("SIGNSPEAK - Live Noon (ن) Detection")
print("=" * 60)

MODEL_DIR  = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_noon"
MODEL_PATH = os.path.join(MODEL_DIR, "noon_robust.tflite")
INFO_PATH  = os.path.join(MODEL_DIR, "noon_robust_info.json")

# Load threshold chosen during training
THRESHOLD = 0.90
if os.path.exists(INFO_PATH):
    with open(INFO_PATH,'r',encoding='utf-8') as f:
        info = json.load(f)
    THRESHOLD = max(float(info.get('threshold', 0.90)), 0.85)
    print(f"   Using trained threshold: {THRESHOLD:.2f}")

SMOOTHING_WINDOW     = 10
REQUIRED_CONSECUTIVE = 10    # stricter = only real Noon passes

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}\n"
                            f"   Run noon_train.py first.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det, out_det = interpreter.get_input_details(), interpreter.get_output_details()

# ---------- Extra geometrical gate ----------
# Even if the model says "Noon", we only accept if the pose ALSO
# looks like Noon structurally. This kills false positives.
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP,   RING_PIP   = 16, 14
PINKY_TIP,  PINKY_PIP  = 20, 18
WRIST = 0

def geom_is_noon(pts):
    """
    pts: list of 21 (x,y) tuples.
    Noon pose: index extended downward, others curled.
    """
    def dist(a,b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    idx_ext   = dist(pts[INDEX_TIP], pts[WRIST]) > dist(pts[INDEX_PIP], pts[WRIST]) * 1.05
    mid_curl  = dist(pts[MIDDLE_TIP], pts[WRIST]) < dist(pts[MIDDLE_PIP], pts[WRIST]) * 1.05
    ring_curl = dist(pts[RING_TIP],   pts[WRIST]) < dist(pts[RING_PIP],   pts[WRIST]) * 1.05
    pink_curl = dist(pts[PINKY_TIP],  pts[WRIST]) < dist(pts[PINKY_PIP],  pts[WRIST]) * 1.05

    # Index tip should be BELOW the wrist (pointing down)
    idx_down = pts[INDEX_TIP][1] > pts[WRIST][1]

    return idx_ext and mid_curl and ring_curl and pink_curl and idx_down

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_sty   = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("\n🎥 q = quit")

pred_history = []
streak = 0

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)
    res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    disp  = frame.copy()
    hand_found = False
    smoothed   = 0.0
    geo_ok     = False

    if res.multi_hand_landmarks:
        hand_found = True
        lm = res.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            disp, lm, mp_hands.HAND_CONNECTIONS,
            mp_sty.get_default_hand_landmarks_style(),
            mp_sty.get_default_hand_connections_style()
        )

        pts = [(p.x, p.y) for p in lm.landmark]
        geo_ok = geom_is_noon(pts)

        feats = []
        for x, y in pts:
            feats += [x, y]
        feats = np.array(feats, dtype=np.float32).reshape(1, 42)

        interpreter.set_tensor(in_det[0]['index'], feats)
        interpreter.invoke()
        pred = float(interpreter.get_tensor(out_det[0]['index'])[0][0])

        pred_history.append(pred)
        if len(pred_history) > SMOOTHING_WINDOW:
            pred_history.pop(0)
        smoothed = float(np.mean(pred_history))

    # Accept only if BOTH the model AND the geometry agree
    accept_now = hand_found and smoothed >= THRESHOLD and geo_ok
    streak = streak + 1 if accept_now else 0
    detected = streak >= REQUIRED_CONSECUTIVE

    # ---------- UI ----------
    if hand_found:
        col = (0,255,0) if detected else (0,180,255)
        cv2.putText(disp, f"Noon score: {smoothed:.3f}", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(disp, f"Streak: {streak}/{REQUIRED_CONSECUTIVE}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.putText(disp, f"Geometry: {'OK' if geo_ok else 'FAIL'}",
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if geo_ok else (0,0,255), 2)

        if detected:
            cv2.rectangle(disp, (5,5),
                          (disp.shape[1]-5, disp.shape[0]-5),
                          (0,255,0), 4)
            cv2.putText(disp, "NOON (ن)", (10,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4)
        else:
            cv2.putText(disp, "Not Noon", (10,170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 3)
    else:
        pred_history.clear(); streak = 0
        cv2.putText(disp, "No hand", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("SignSpeak - Noon (ن)", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows(); hands.close()
print("\n✅ Stopped.")