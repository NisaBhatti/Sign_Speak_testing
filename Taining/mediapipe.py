# generate_hand_data.py
import cv2
import mediapipe as mp
import numpy as np
import os
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_landmarks(image_path):
    """Extract 21 hand landmarks from image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
    
    return landmarks

# Collect training data
def create_training_data(image_folder, output_file='hand_data.json'):
    """Create training data from images"""
    data = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(image_folder, filename)
            landmarks = extract_landmarks(path)
            if landmarks:
                data.append({
                    'image': filename,
                    'landmarks': landmarks
                })
                print(f'✅ Processed: {filename}')
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f'✅ Saved {len(data)} samples to {output_file}')
    return data

if __name__ == '__main__':
    create_training_data('hand_images/')