import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Load images and extract landmarks on the fly
def load_data(image_folder):
    X, y = [], []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(image_folder, filename))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                # Extract 42 landmarks
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y])
                X.append(landmarks)
                
                # Label based on folder name or filename
                label = 1 if 'alif' in filename.lower() else 0
                y.append(label)
    
    return np.array(X), np.array(y)

# Load data
X, y = load_data("path/to/your/images")
print(f"Loaded {len(X)} samples")

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train your model (same as before)
# ... (rest of your training code)