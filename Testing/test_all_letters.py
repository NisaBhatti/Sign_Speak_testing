# ============================================================
# Save as: test_all_letters.py
# Live webcam — one model, all letters.
# Displays:  ENGLISH NAME  +  Arabic letter  +  confidence
# ============================================================

import cv2, os, json
import numpy as np
import tensorflow as tf
import mediapipe as mp

print("=" * 60)
print("SIGNSPEAK - Live All-Letters Detection")
print("=" * 60)

MODEL_DIR  = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\exported_models_all"
MODEL_PATH = os.path.join(MODEL_DIR, "all_letters.tflite")
INFO_PATH  = os.path.join(MODEL_DIR, "all_letters_info.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

with open(INFO_PATH, 'r', encoding='utf-8') as f:
    info = json.load(f)

arabic_letters = info['arabic_letters']
english_names  = info['english_names']
NUM_CLASSES    = info['num_classes']

print(f"\n   {NUM_CLASSES} classes loaded:")
for en, ar in zip(english_names, arabic_letters):
    print(f"      {en:10s}  {ar}")

# ---------- Load TFLite ----------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()
out_det = interpreter.get_output_details()

# ---------- Config ----------
CONF_THRESHOLD       = 0.80
REQUIRED_CONSECUTIVE = 8
SMOOTHING_WINDOW     = 10

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_sty   = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ---------- Camera ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
print("\n🎥 q = quit\n")

prob_history = []
last_class   = None
streak       = 0

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)
    res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    disp  = frame.copy()
    hand_found = False

    if res.multi_hand_landmarks:
        hand_found = True
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            disp, lm, mp_hands.HAND_CONNECTIONS,
            mp_sty.get_default_hand_landmarks_style(),
            mp_sty.get_default_hand_connections_style()
        )

        # Extract 42 features — same order as training
        feats = []
        for p in lm.landmark:
            feats += [p.x, p.y]
        feats = np.array(feats, dtype=np.float32).reshape(1, 42)
        feats = np.clip(feats, 0, 1)

        interpreter.set_tensor(in_det[0]['index'], feats)
        interpreter.invoke()
        probs = interpreter.get_tensor(out_det[0]['index'])[0]

        prob_history.append(probs)
        if len(prob_history) > SMOOTHING_WINDOW:
            prob_history.pop(0)
        smooth = np.mean(prob_history, axis=0)

        top_idx  = int(np.argmax(smooth))
        top_conf = float(smooth[top_idx])

        # ---------- Streak logic ----------
        if top_conf >= CONF_THRESHOLD and top_idx == last_class:
            streak += 1
        elif top_conf >= CONF_THRESHOLD:
            streak = 1
            last_class = top_idx
        else:
            streak = 0
            last_class = None

        stable = streak >= REQUIRED_CONSECUTIVE

        # ================= UI =================
        color = (0, 255, 0) if stable else (0, 180, 255)

        # Top line — English + Arabic + conf
        cv2.putText(disp,
                    f"Top: {english_names[top_idx]} ({arabic_letters[top_idx]})   "
                    f"conf={top_conf:.3f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(disp, f"streak: {streak}/{REQUIRED_CONSECUTIVE}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Top-3 list with names
        top3 = np.argsort(smooth)[::-1][:3]
        for rank, i in enumerate(top3):
            y = 105 + rank * 30
            text = (f"{rank+1}.  {english_names[i]:<8s} {arabic_letters[i]}"
                    f"   {smooth[i]:.2f}")
            cv2.putText(disp, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)

        # Big centered output when stable
        if stable:
            cv2.rectangle(disp, (5, 5),
                          (disp.shape[1] - 5, disp.shape[0] - 5),
                          (0, 255, 0), 4)

            # Big Arabic letter (center)
            ar = arabic_letters[top_idx]
            (tw, th), _ = cv2.getTextSize(ar, cv2.FONT_HERSHEY_SIMPLEX, 5.0, 8)
            cv2.putText(disp, ar,
                        (disp.shape[1] // 2 - tw // 2, disp.shape[0] // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 8)

            # English name under it
            en = english_names[top_idx]
            (tw2, th2), _ = cv2.getTextSize(en, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)
            cv2.putText(disp, en,
                        (disp.shape[1] // 2 - tw2 // 2, disp.shape[0] // 2 + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)

            # Confidence under name
            conf_txt = f"{top_conf * 100:.1f}%"
            (tw3, th3), _ = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(disp, conf_txt,
                        (disp.shape[1] // 2 - tw3 // 2, disp.shape[0] // 2 + 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "Detecting...", (10, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        prob_history.clear()
        streak = 0
        last_class = None
        cv2.putText(disp, "No hand detected", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("SignSpeak - All Letters", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows(); hands.close()
print("\n✅ Stopped.")