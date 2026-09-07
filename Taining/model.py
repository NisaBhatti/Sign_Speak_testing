# ============================================
# Using MediaPipe Pre-trained Model
# ============================================
# mediapipe_hand_detection.py

import cv2
import mediapipe as mp
import numpy as np

class MediaPipeHandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, image):
        """Detect hands in image"""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.hands.process(rgb_image)
        
        # Draw results
        annotated_image = image.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_image, results
    
    def extract_landmarks(self, results):
        """Extract landmark coordinates"""
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_lm = []
                for lm in hand_landmarks.landmark:
                    hand_lm.append([lm.x, lm.y, lm.z])
                landmarks.append(hand_lm)
        return landmarks

# Test MediaPipe detection
detector = MediaPipeHandDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated_frame, results = detector.detect(frame)
    landmarks = detector.extract_landmarks(results)
    
    cv2.imshow('Hand Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()