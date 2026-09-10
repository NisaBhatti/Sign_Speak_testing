# Save as: create_better_tay_data.py

import cv2
import numpy as np
import mediapipe as mp
import os
import shutil

print("="*70)
print("CREATE BETTER TAY TRAINING DATA")
print("="*70)

SOURCE_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\Tay"
OUTPUT_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset\Tay_Enhanced"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def augment_image(img, landmarks):
    """Create augmented versions of an image with hand landmarks"""
    augmented = []
    
    # 1. Original
    augmented.append(img.copy())
    
    # 2. Flip horizontally
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    # 3. Rotate slightly
    h, w = img.shape[:2]
    center = (w//2, h//2)
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)
    
    # 4. Brightness variations
    for alpha in [0.7, 0.85, 1.15, 1.3]:
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(bright)
    
    # 5. Contrast variations
    for alpha in [0.8, 1.2]:
        contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=20)
        augmented.append(contrast)
    
    return augmented

print(f"📁 Source: {SOURCE_PATH}")
print(f"📁 Output: {OUTPUT_PATH}")

# Get images
image_files = []
for ext in ['.jpg', '.jpeg', '.png']:
    image_files.extend([f for f in os.listdir(SOURCE_PATH) if f.lower().endswith(ext)])

print(f"\n📊 Found {len(image_files)} images")

total_created = 0
valid_images = 0

for idx, filename in enumerate(image_files):
    filepath = os.path.join(SOURCE_PATH, filename)
    img = cv2.imread(filepath)
    
    if img is None:
        continue
    
    # Check if hand is detected
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        continue
    
    valid_images += 1
    
    # Augment
    augmented_images = augment_image(img, results.multi_hand_landmarks[0].landmark)
    
    # Save augmented versions
    name, ext = os.path.splitext(filename)
    for i, aug_img in enumerate(augmented_images):
        aug_filename = f"{name}_aug_{i:03d}{ext}"
        aug_path = os.path.join(OUTPUT_PATH, aug_filename)
        cv2.imwrite(aug_path, aug_img)
        total_created += 1
    
    if (idx + 1) % 5 == 0:
        print(f"   Processed {idx + 1}/{len(image_files)} - Created {total_created} images")

hands.close()

print(f"\n✅ Valid images found: {valid_images}")
print(f"✅ Total images created: {total_created}")
print(f"✅ Output folder: {OUTPUT_PATH}")

print("\n📝 Update your training script to use:")
print(f"   DATASET_PATH = r\"{OUTPUT_PATH}\"")
print("\n✅ Done!")