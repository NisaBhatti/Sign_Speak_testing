"""
PSL Data Collection Script
Collect hand sign images for training
Press 's' to save images, 'q' to quit
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Configuration
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20  # Padding around hand
imgSize = 300  # Square size for normalized images

# CHANGE THIS for each sign you want to collect
SIGN_NAME = "Alif"  # Change to: Bay, Pay, etc.
folder = f"Data/{SIGN_NAME}"
os.makedirs(folder, exist_ok=True)

counter = 0

print(f"Collecting images for: {SIGN_NAME}")
print("Press 's' to save image, 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break
    
    # Find hands
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop hand with padding
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size > 0:
            aspectRatio = h / w
            
            try:
                if aspectRatio > 1:
                    # Tall image
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    # Wide image
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                
                cv2.imshow("Cropped Hand", imgCrop)
                cv2.imshow("Normalized Hand", imgWhite)
                
            except Exception as e:
                print(f"Resize error: {e}")
    
    cv2.imshow("Camera", img)
    key = cv2.waitKey(1)
    
    if key == ord("s") and hands:
        counter += 1
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved {filename} | Count: {counter}")
    
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Collected {counter} images for {SIGN_NAME}")
