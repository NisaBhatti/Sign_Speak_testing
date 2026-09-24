# Save as: model_training/test_bay_live.py
# STRICT Bay (ب) live detection - tilt-tolerant geometric rule

import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import os
import sys

print("=" * 60)
print("SIGNSPEAK - STRICT Bay (ب) Live Detection")
print("=" * 60)

# ========== YOUR PC PATH ==========
MODEL_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_bay\bay_robust.tflite"
NEURAL_THRESHOLD = 0.85   # neural net must be this confident
USE_GEOMETRIC_RULE = True # AND the geometric rule must pass

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("   Run train_bay_robust.py first.")
    sys.exit(1)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Model input shape: {input_details[0]['shape']}")
print(f"   Neural threshold:  {NEURAL_THRESHOLD}")
print(f"   Geometric rule:    {'ON' if USE_GEOMETRIC_RULE else 'OFF'}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ========== SAME TILT-TOLERANT RULE AS TRAINING ==========
def is_strict_bay_pose(landmarks_42, verbose=False):
    """
    Bay sign: 4 fingers extended (distance from wrist increases),
    fingers close together, thumb bent across palm, hand upright.
    """
    lm = landmarks_42.reshape(21, 2)

    wrist = lm[0]
    thumb_tip = lm[4]; thumb_mcp = lm[2]
    index_mcp = lm[5];  index_pip = lm[6];  index_dip = lm[7];  index_tip = lm[8]
    middle_mcp = lm[9]; middle_pip = lm[10]; middle_dip = lm[11]; middle_tip = lm[12]
    ring_mcp = lm[13];  ring_pip = lm[14];  ring_dip = lm[15];  ring_tip = lm[16]
    pinky_mcp = lm[17]; pinky_pip = lm[18]; pinky_dip = lm[19]; pinky_tip = lm[20]

    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def is_extended(mcp, pip, dip, tip):
        d_mcp = dist(mcp, wrist)
        d_pip = dist(pip, wrist)
        d_dip = dist(dip, wrist)
        d_tip = dist(tip, wrist)
        return (d_tip > d_mcp + 0.05) and (d_dip > d_pip - 0.02) and (d_pip > d_mcp - 0.02)

    # RULE 1: all 4 fingers extended
    index_up  = is_extended(index_mcp, index_pip, index_dip, index_tip)
    middle_up = is_extended(middle_mcp, middle_pip, middle_dip, middle_tip)
    ring_up   = is_extended(ring_mcp, ring_pip, ring_dip, ring_tip)
    pinky_up  = is_extended(pinky_mcp, pinky_pip, pinky_dip, pinky_tip)

    if not (index_up and middle_up and ring_up and pinky_up):
        if verbose:
            print(f"    Rule1 fail: idx={index_up} mid={middle_up} ring={ring_up} pink={pinky_up}")
        return False, "Not all 4 fingers extended"

    # RULE 2: fingers not spread wide
    tip_xs = [index_tip[0], middle_tip[0], ring_tip[0], pinky_tip[0]]
    spread = max(tip_xs) - min(tip_xs)
    if spread > 0.35:
        if verbose:
            print(f"    Rule2 fail: spread={spread:.3f}")
        return False, f"Fingers spread ({spread:.2f})"

    # RULE 3: thumb bent
    thumb_to_index_mcp = dist(thumb_tip, index_mcp)
    if thumb_to_index_mcp > 0.22:
        if verbose:
            print(f"    Rule3 fail: thumb dist={thumb_to_index_mcp:.3f}")
        return False, f"Thumb not bent ({thumb_to_index_mcp:.2f})"

    if dist(thumb_tip, wrist) > dist(thumb_mcp, wrist) + 0.10:
        if verbose:
            print(f"    Rule3b fail: thumb extended")
        return False, "Thumb extended"

    # RULE 4: hand upright
    dx_total = 0.0
    dy_total = 0.0
    for mcp, tip in [(index_mcp, index_tip), (middle_mcp, middle_tip),
                     (ring_mcp, ring_tip), (pinky_mcp, pinky_tip)]:
        dx_total += (tip[0] - mcp[0])
        dy_total += (tip[1] - mcp[1])

    avg_dx = dx_total / 4.0
    avg_dy = dy_total / 4.0

    if avg_dy > -0.05:
        if verbose:
            print(f"    Rule4 fail: avg_dy={avg_dy:.3f} (not pointing up)")
        return False, "Hand not upright"

    if abs(avg_dx) > abs(avg_dy):
        if verbose:
            print(f"    Rule4b fail: too sideways dx={avg_dx:.3f} dy={avg_dy:.3f}")
        return False, "Hand sideways"

    return True, "OK"


def predict_bay(landmarks_42):
    inp = np.array([landmarks_42], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]['index'])[0][0])


def landmarks_to_features(hl):
    f = []
    for lm in hl.landmark:
        f.append(lm.x)
        f.append(lm.y)
    return np.array(f, dtype=np.float32)


# ========== WEBCAM LOOP ==========
print(f"\n🎥 Starting webcam... (Press 'q' to quit)")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)

    status_text = "No Hand Detected"
    status_color = (128, 128, 128)
    confidence = 0.0
    is_bay = False
    rule_ok = False
    rule_reason = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

        features = landmarks_to_features(hand_landmarks)
        confidence = predict_bay(features)
        rule_ok, rule_reason = is_strict_bay_pose(features)

        # STRICT: require BOTH
        if USE_GEOMETRIC_RULE:
            is_bay = rule_ok and (confidence >= NEURAL_THRESHOLD)
        else:
            is_bay = confidence >= NEURAL_THRESHOLD

        if is_bay:
            status_text = f"BAY (ب) DETECTED: {confidence*100:.1f}%"
            status_color = (0, 255, 0)
        else:
            if not rule_ok:
                status_text = f"NOT BAY - {rule_reason}"
            else:
                status_text = f"NOT BAY - Low confidence ({confidence*100:.1f}%)"
            status_color = (0, 0, 255)

    # Status banner
    cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # Confidence bar
    bar_width = int((w - 30) * confidence)
    bar_color = (0, 255, 0) if is_bay else (0, 0, 255)
    cv2.rectangle(frame, (15, h - 40), (15 + bar_width, h - 15), bar_color, -1)
    cv2.rectangle(frame, (15, h - 40), (w - 15, h - 15), (255, 255, 255), 2)

    # Info line
    info = f"NN: {confidence*100:.1f}%  |  Rule: {'PASS' if rule_ok else 'FAIL'}"
    cv2.putText(frame, info, (15, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Threshold marker on bar
    thr_x = 15 + int((w - 30) * NEURAL_THRESHOLD)
    cv2.line(frame, (thr_x, h - 45), (thr_x, h - 10), (255, 255, 0), 2)

    cv2.imshow("SignSpeak - STRICT Bay (ب)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("\n✅ Testing complete.")