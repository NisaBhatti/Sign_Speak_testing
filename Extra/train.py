"""
PSL Model Training Script for Existing Dataset
Compatible with cvzone's Classifier (same as the working repository)
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cvzone.HandTrackingModule import HandDetector
import math
import warnings
warnings.filterwarnings('ignore')

print("🇵🇰 PSL Model Training (Repository-Compatible)")
print("="*60)

# ==================== CONFIGURATION ====================
# ⚠️ CHANGE THIS TO YOUR DATASET PATH
DATASET_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset"

# Your dataset structure should be:
# Simple_Dataset/
#   ├── Alif/
#   │   ├── image1.jpg
#   │   ├── image2.jpg
#   │   └── ...
#   ├── Bay/
#   ├── Pay/
#   └── ...

# Image preprocessing (same as repository)
IMG_SIZE = 300  # Same as in their test.ipynb
OFFSET = 20     # Padding around hand

# Training settings
BATCH_SIZE = 32
EPOCHS = 30
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.15

# ==================== STEP 1: LOAD AND PREPROCESS DATA ====================
print("\n📂 Step 1: Loading and preprocessing dataset...")
print("-"*40)

# Initialize hand detector for preprocessing
detector = HandDetector(maxHands=1, detectionCon=0.8)

def normalize_hand_image(image_path):
    """
    Process image exactly like the repository's data collection:
    - Detect hand
    - Crop with padding
    - Resize to 300x300 with white background
    Returns processed image or None if no hand detected
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Detect hand
        hands, _ = detector.findHands(img, draw=False)
        
        if not hands:
            return None
        
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Create white background
        imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        
        # Crop hand with offset padding
        y1 = max(0, y - OFFSET)
        y2 = min(img.shape[0], y + h + OFFSET)
        x1 = max(0, x - OFFSET)
        x2 = min(img.shape[1], x + w + OFFSET)
        
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size == 0:
            return None
        
        # Resize and place on white background (same logic as repository)
        aspectRatio = h / w
        
        if aspectRatio > 1:
            # Tall image
            k = IMG_SIZE / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
            wGap = math.ceil((IMG_SIZE - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # Wide image
            k = IMG_SIZE / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
            hGap = math.ceil((IMG_SIZE - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
        return imgWhite
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Get all sign classes (folder names)
sign_classes = [d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d))]
sign_classes.sort()

print(f"✅ Found {len(sign_classes)} sign classes:")
for i, sign in enumerate(sign_classes[:10]):
    folder_path = os.path.join(DATASET_PATH, sign)
    img_count = len([f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"   {i+1}. {sign}: {img_count} images")
if len(sign_classes) > 10:
    print(f"   ... and {len(sign_classes)-10} more")

# Load and process all images
X = []  # Processed images
y = []  # Labels
failed_count = 0

print("\n🖐️ Processing images (this may take a few minutes)...")

for label_idx, sign_name in enumerate(sign_classes):
    folder_path = os.path.join(DATASET_PATH, sign_name)
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"   Processing {sign_name}: ", end="")
    success_count = 0
    
    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        processed_img = normalize_hand_image(img_path)
        
        if processed_img is not None:
            X.append(processed_img)
            y.append(label_idx)
            success_count += 1
        else:
            failed_count += 1
    
    print(f"✅ {success_count}/{len(images)}")

print(f"\n📊 Processing complete!")
print(f"   Valid images: {len(X)}")
print(f"   Failed (no hand detected): {failed_count}")

if len(X) == 0:
    print("\n❌ ERROR: No valid images found! Check your dataset path and images.")
    exit()

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"   Image shape: {X[0].shape}")
print(f"   Classes: {len(sign_classes)}")

# ==================== STEP 2: SPLIT DATA ====================
print("\n📊 Step 2: Splitting data...")
print("-"*40)

# First split: training+validation vs test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

# Second split: training vs validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, 
    test_size=VALIDATION_SPLIT, 
    random_state=42, 
    stratify=y_train_val
)

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

print(f"   Training: {len(X_train)} images")
print(f"   Validation: {len(X_val)} images")
print(f"   Testing: {len(X_test)} images")

# Convert labels to categorical
y_train_cat = keras.utils.to_categorical(y_train, num_classes=len(sign_classes))
y_val_cat = keras.utils.to_categorical(y_val, num_classes=len(sign_classes))
y_test_cat = keras.utils.to_categorical(y_test, num_classes=len(sign_classes))

# ==================== STEP 3: DATA AUGMENTATION ====================
print("\n🔄 Step 3: Setting up data augmentation...")
print("-"*40)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    shear_range=0.1,
    horizontal_flip=False  # Keep false for sign language (hand orientation matters)
)

# ==================== STEP 4: BUILD MODEL ====================
print("\n🏗️ Step 4: Building model...")
print("-"*40)

# Using MobileNetV2 (same architecture that works well for PSL)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers initially
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(sign_classes), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==================== STEP 5: TRAIN MODEL ====================
print("\n🔥 Step 5: Training model...")
print("-"*40)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_psl_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ==================== STEP 6: EVALUATE ====================
print("\n📈 Step 6: Evaluating model...")
print("-"*40)

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# Fine-tuning (optional - uncomment to improve accuracy)
print("\n🔧 Optional: Fine-tuning model...")
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
history_finetune = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val_cat),
    epochs=fine_tune_epochs,
    callbacks=callbacks,
    verbose=1
)

# ==================== STEP 7: SAVE MODEL (Repository Compatible) ====================
print("\n💾 Step 7: Saving model for real-time detection...")
print("-"*40)

# Create Model folder (same structure as repository)
os.makedirs("Model", exist_ok=True)

# Save Keras model
model.save("Model/keras_model.h5")
print(f"✅ Saved: Model/keras_model.h5")

# Save labels in the exact format cvzone expects
with open("Model/labels.txt", "w", encoding='utf-8') as f:
    for i, sign in enumerate(sign_classes):
        f.write(f"{i} {sign}\n")

print(f"✅ Saved: Model/labels.txt")

# Also save a copy with full class mapping for reference
with open("Model/class_mapping.txt", "w", encoding='utf-8') as f:
    f.write("PSL Class Index Mapping\n")
    f.write("="*30 + "\n")
    for i, sign in enumerate(sign_classes):
        f.write(f"{i}: {sign}\n")

# Convert to TFLite for Flutter app
print("\n📱 Converting to TFLite for Flutter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

try:
    tflite_model = converter.convert()
    with open("Model/psl_model.tflite", "wb") as f:
        f.write(tflite_model)
    print(f"✅ Saved: Model/psl_model.tflite")
    
    # File sizes
    import os
    keras_size = os.path.getsize("Model/keras_model.h5") / (1024 * 1024)
    tflite_size = os.path.getsize("Model/psl_model.tflite") / (1024 * 1024)
    print(f"   Keras model: {keras_size:.2f} MB")
    print(f"   TFLite model: {tflite_size:.2f} MB")
    
except Exception as e:
    print(f"⚠️ TFLite conversion warning: {e}")

# ==================== STEP 8: VISUALIZE RESULTS ====================
print("\n📊 Step 8: Generating visualizations...")
print("-"*40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Loss plot
axes[1].plot(history.history['loss'], label='Training', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Model/training_history.png", dpi=100)
plt.show()

print(f"✅ Saved: Model/training_history.png")

# ==================== STEP 9: CREATE TEST SCRIPT ====================
print("\n🧪 Step 9: Creating test script...")
print("-"*40)

test_script = '''"""
PSL Real-time Detection Test Script
Run this to test your trained model
Usage: python test_detection.py
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

print("🚀 PSL Detection Started")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Prepare image for classification
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop hand
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size > 0:
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
            
            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = int(prediction[index] * 100)
            
            # Read label from file
            with open("Model/labels.txt", "r") as f:
                labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]
            
            # Draw results
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                         (x - offset + 250, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, f"{labels[index]} {confidence}%", (x, y - 26),
                       cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                         (x + w + offset, y + h + offset), (255, 0, 255), 4)
            
            cv2.imshow("Cropped Hand", imgCrop)
            cv2.imshow("Normalized Hand", imgWhite)
    
    cv2.imshow("Pakistan Sign Language Detection", imgOutput)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

with open("test_detection.py", "w") as f:
    f.write(test_script)

print(f"✅ Created: test_detection.py")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE! 🎉")
print("="*60)

print(f"\n✅ Model saved in 'Model/' folder:")
print(f"   • keras_model.h5 - For cvzone real-time detection")
print(f"   • labels.txt - Class labels (cvzone format)")
print(f"   • psl_model.tflite - For Flutter app integration")
print(f"   • training_history.png - Accuracy graph")

print(f"\n📊 Final Performance:")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

print(f"\n🚀 Next Steps:")
print(f"   1. Run real-time test: python test_detection.py")
print(f"   2. For Flutter: Copy 'psl_model.tflite' and 'labels.txt' to your app")
print(f"   3. Deploy to Flutter using tflite_flutter package")

print(f"\n💡 Tips if accuracy is low:")
print(f"   • Collect more images per sign (aim for 100+ per sign)")
print(f"   • Ensure hands are clearly visible with good lighting")
print(f"   • Add more varied backgrounds during data collection")
print(f"   • Run fine-tuning again with more epochs")