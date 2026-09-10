# Save as: D:\MODEL\Sign_Speak_testing-main\Testing\test_tay_robust.py
# v2 — with geometric gate that REJECTS Alif-like poses

import os
import cv2
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("=" * 60)
print("SIGNSPEAK - Tay Detection (with Alif rejection gate)")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================
BASE_DIR   = r"D:\MODEL\Sign_Speak_testing-main"
MODEL_PATH = os.path.join(BASE_DIR, "Exported_Model", "tay_robust.tflite")
INFO_PATH  = os.path.join(BASE_DIR, "Exported_Model", "tay_robust_info.json")

# ============================================================
# LOAD MODEL + INFO
# ============================================================
print("\n📁 Loading model...")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun tay_train.py first.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

# Use the trained threshold, but force at least 0.75 (strict)
NN_THRESHOLD = max(float(info.get("threshold", 0.5)), 0.75)
print(f"   Model     : {info.get('model', 'tay_robust')}")
print(f"   Sign      : {info.get('sign', 'ط')}  ({info.get('sign_name', 'Tay')})")
print(f"   NN Threshold (strict): {NN_THRESHOLD}")

# ============================================================
# NORMALIZATION  (MUST MATCH TRAINING)
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
# GEOMETRIC GATE — rejects Alif-like poses
# ============================================================
#
# MediaPipe landmark indices:
#   0 = wrist
#   5 = index MCP (base of index finger)
#   8 = index TIP
#   12 = middle TIP
#   16 = ring TIP
#   20 = pinky TIP
#
# Alif (ا): index finger points UP → tip has SMALLER y than MCP
#           middle/ring/pinky folded → their tips are close to their MCPs
# Tay (ط):  index finger points SIDEWAYS → tip has similar y to MCP
#           but large Δx
#           middle/ring/pinky folded

def tay_geometric_gate(hand_landmarks):
    """
    Returns (is_tay_pose, diagnostics_dict).
    Uses raw (unnormalized) mediapipe landmarks — that's fine for direction checks
    because direction is scale-invariant.
    """
    lm = hand_landmarks.landmark

    # Point helper: returns (x, y)
    def P(i):
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    wrist      = P(0)
    idx_mcp    = P(5)
    idx_tip    = P(8)
    mid_mcp    = P(9)
    mid_tip    = P(12)
    ring_mcp   = P(13)
    ring_tip   = P(16)
    pinky_mcp  = P(17)
    pinky_tip  = P(20)

    # --- Feature 1: index finger direction ---
    vec = idx_tip - idx_mcp              # from base to tip
    vx, vy = float(vec[0]), float(vec[1])

    # angle from vertical axis (y). |vx| small + |vy| large → vertical (Alif)
    # |vx| large + |vy| small → horizontal (Tay)
    angle_from_vertical = np.degrees(np.arctan2(abs(vx), abs(vy) + 1e-6))

    # --- Feature 2: index finger straightness ---
    idx_pip = P(6)                       # index PIP
    idx_dip = P(7)                       # index DIP
    idx_seg1 = idx_pip - idx_mcp
    idx_seg2 = idx_tip - idx_dip
    idx_straightness = float(
        np.dot(idx_seg1, idx_seg2) /
        (np.linalg.norm(idx_seg1) * np.linalg.norm(idx_seg2) + 1e-6)
    )

    # --- Feature 3: other fingers folded? ---
    def finger_fold_ratio(mcp, tip):
        d_tip = np.linalg.norm(tip - wrist)
        d_mcp = np.linalg.norm(mcp - wrist)
        return d_tip / (d_mcp + 1e-6)

    mid_fold   = finger_fold_ratio(mid_mcp,   mid_tip)
    ring_fold  = finger_fold_ratio(ring_mcp,  ring_tip)
    pinky_fold = finger_fold_ratio(pinky_mcp, pinky_tip)

    avg_other_fold = (mid_fold + ring_fold + pinky_fold) / 3.0

    # --- Decision rules ---
    # Alif: index points mostly vertical  → angle_from_vertical < 35°
    # Tay : index points mostly sideways  → angle_from_vertical > 45°
    is_horizontal   = angle_from_vertical > 45.0
    is_straight     = idx_straightness > 0.85
    others_folded   = avg_other_fold < 1.35

    # Additional reject condition: index tip is HIGH above wrist
    # (that's Alif — vertical finger reaches up)
    wrist_to_tip_y = idx_tip[1] - wrist[1]
    tip_above_wrist = wrist_to_tip_y < -0.20   # negative y = up in image coords

    # Tay requires: horizontal AND others folded AND NOT tip way above wrist
    is_tay_pose = (
        is_horizontal
        and others_folded
        and (not tip_above_wrist)
    )

    diag = {
        "angle_from_vertical": float(angle_from_vertical),
        "index_straightness":  float(idx_straightness),
        "avg_other_fold":      float(avg_other_fold),
        "tip_above_wrist":     bool(tip_above_wrist),
        "is_horizontal":       bool(is_horizontal),
        "others_folded":       bool(others_folded),
    }
    return is_tay_pose, diag

# ============================================================
# MEDIAPIPE
# ============================================================
print("\n🖐️ Starting MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# CAMERA
# ============================================================
print("\n📷 Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WIN = "Tay Detection — Alif-Rejecting"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1100, 800)

print("\n" + "=" * 60)
print("🎯 TAY DETECTION  (Alif is explicitly rejected)")
print("=" * 60)
print("   Tay (ط) = index finger SIDEWAYS, other fingers folded")
print("   Alif (ا) = index finger UP → automatically rejected")
print("")
print("   Q=Quit  S=Save  D=Toggle diagnostics")
print("=" * 60 + "\n")

# ============================================================
# STATE
# ============================================================
nn_history     = deque(maxlen=10)
gate_history   = deque(maxlen=10)
frame_count    = 0
show_debug     = True

def get_nn_features(hand_landmarks):
    feats = []
    for lm in hand_landmarks.landmark:
        feats.extend([lm.x, lm.y])
    raw = np.array(feats, dtype=np.float32)
    return normalize_landmarks(raw).reshape(1, -1)

def predict_nn(features):
    interpreter.set_tensor(input_details[0]["index"], features)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]["index"])[0][0])

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # ---- Draw hand ----
            for lm in hand_landmarks.landmark:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 6, (0, 255, 0), -1)
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 8, (0, 200, 0), 2)
            for c in mp_hands.HAND_CONNECTIONS:
                s = hand_landmarks.landmark[c[0]]
                e = hand_landmarks.landmark[c[1]]
                cv2.line(frame,
                         (int(s.x*w), int(s.y*h)),
                         (int(e.x*w), int(e.y*h)),
                         (0, 255, 0), 2)

            # ---- Stage 1: neural network ----
            feats = get_nn_features(hand_landmarks)
            nn_score = predict_nn(feats)
            nn_history.append(nn_score)
            nn_smooth = sum(nn_history) / len(nn_history)

            nn_says_tay = nn_smooth > NN_THRESHOLD

            # ---- Stage 2: geometric gate (only runs if NN said yes) ----
            gate_says_tay, diag = tay_geometric_gate(hand_landmarks)

            gate_history.append(1.0 if gate_says_tay else 0.0)
            gate_smooth = sum(gate_history) / len(gate_history)

            # Final decision: BOTH stages must agree
            final_tay = nn_says_tay and (gate_smooth > 0.5)

            # ---- Header ----
            cv2.rectangle(frame, (0, 0), (w, 130), (0, 0, 0), -1)

            if final_tay:
                text, color = "✅ TAY (ط) — VERIFIED", (0, 255, 0)
            elif nn_says_tay and not gate_says_tay:
                text, color = "❌ REJECTED — looks like ALIF, not Tay", (0, 128, 255)
            elif nn_smooth > 0.4:
                text, color = "⚠️ UNCERTAIN", (0, 165, 255)
            else:
                text, color = "❌ NOT TAY", (0, 0, 255)

            cv2.putText(frame, text, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame,
                        f"NN: {nn_smooth*100:.1f}% (thr {NN_THRESHOLD:.2f})  "
                        f"Gate: {'PASS' if gate_says_tay else 'FAIL'}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            # ---- Diagnostics panel ----
            if show_debug:
                cv2.putText(frame,
                            f"angle_from_vert: {diag['angle_from_vertical']:.1f}°   "
                            f"(>45° = horizontal)",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                cv2.putText(frame,
                            f"straightness: {diag['index_straightness']:.2f}   "
                            f"others_folded: {diag['avg_other_fold']:.2f}   "
                            f"tip_above_wrist: {diag['tip_above_wrist']}",
                            (20, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # Rejection reason
                if nn_says_tay and not gate_says_tay:
                    if not diag["is_horizontal"]:
                        reason = "index finger too VERTICAL (looks like Alif)"
                    elif diag["tip_above_wrist"]:
                        reason = "index tip above wrist (looks like Alif)"
                    elif not diag["others_folded"]:
                        reason = "other fingers not folded"
                    else:
                        reason = "geometry does not match Tay"
                    cv2.putText(frame, f"→ {reason}", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1)

            # ---- Bottom legend ----
            cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "Q=Quit  S=Save  D=Toggle debug",
                        (20, h-13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    else:
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, "Show your hand to camera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        nn_history.clear()
        gate_history.clear()

    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        fn = f"tay_test_{frame_count}.png"
        cv2.imwrite(fn, frame)
        print(f"📸 Saved: {fn}")
    elif key == ord("d"):
        show_debug = not show_debug

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"\n✅ Done! Processed {frame_count} frames")