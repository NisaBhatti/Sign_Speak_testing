"""
PSL (Pakistani Sign Language) Recognition System
This script uses MediaPipe for hand landmark extraction + a custom CNN
Perfect for static signs with high accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("🇵🇰 PSL (Pakistani Sign Language) Training System")
print("="*60)

# ====================== CONFIGURATION ======================
# ⚠️ CHANGE THIS TO YOUR DATASET PATH
DATASET_PATH = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset"

# Your dataset structure should be:
# Simple_Dataset/
#   ├── الف/     (Alif)
#   ├── بے/      (Bay)
#   ├── پے/      (Pay)
#   └── ... etc

IMAGE_SIZE = 224  # Better resolution for hand details
BATCH_SIZE = 32
EPOCHS = 30
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.15  # From training data

# ====================== STEP 1: LOAD AND VERIFY DATASET ======================
print("\n📂 Step 1: Loading Dataset...")
print("-"*40)

# Get all sign folders
sign_classes = [f for f in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, f))]

if len(sign_classes) == 0:
    print("❌ ERROR: No folders found in dataset path!")
    print(f"   Checked path: {DATASET_PATH}")
    exit()

print(f"✅ Found {len(sign_classes)} sign classes:")
for i, sign in enumerate(sign_classes[:10]):  # Show first 10
    folder_path = os.path.join(DATASET_PATH, sign)
    img_count = len([f for f in os.listdir(folder_path) 
                    if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"   {i+1}. {sign}: {img_count} images")
if len(sign_classes) > 10:
    print(f"   ... and {len(sign_classes)-10} more classes")

# ====================== STEP 2: INITIALIZE MEDIAPIPE ======================
print("\n🖐️ Step 2: Initializing Hand Detection...")
print("-"*40)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,  # Important for processing images
    max_num_hands=1,         # We assume one hand per sign
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_landmarks(image_path):
    """
    Extract 21 hand landmarks (x, y, z coordinates) from an image
    Returns: Flattened array of 63 values (21 landmarks × 3 coordinates)
             or None if no hand detected
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert BGR to RGB (MediaPipe expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands_detector.process(img_rgb)
    
    # Check if hand is detected
    if not results.multi_hand_landmarks:
        return None
    
    # Get first hand's landmarks
    hand = results.multi_hand_landmarks[0]
    
    # Extract all landmarks (21 points × 3 coordinates = 63 features)
    landmarks = []
    for landmark in hand.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks)

# ====================== STEP 3: EXTRACT LANDMARKS FROM ALL IMAGES ======================
print("\n🔍 Step 3: Extracting Hand Landmarks...")
print("-"*40)

X_landmarks = []  # Features (landmark coordinates)
y_labels = []     # Labels (sign class names)
failed_images = []

for class_name in sign_classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    images = [f for f in os.listdir(class_path) 
              if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"   Processing {class_name}: ", end="")
    success_count = 0
    
    for img_file in images:
        img_path = os.path.join(class_path, img_file)
        landmarks = extract_hand_landmarks(img_path)
        
        if landmarks is not None:
            X_landmarks.append(landmarks)
            y_labels.append(class_name)
            success_count += 1
        else:
            failed_images.append(img_path)
    
    print(f"✅ {success_count}/{len(images)} successful")

print(f"\n📊 Total Statistics:")
print(f"   ✅ Successful extractions: {len(X_landmarks)}")
print(f"   ❌ Failed extractions: {len(failed_images)}")
print(f"   📁 Total classes: {len(sign_classes)}")

if len(failed_images) > 0:
    print(f"\n⚠️ Warning: {len(failed_images)} images had no detectable hand")
    print("   This is normal if some images are blurry or have bad lighting")

# Check if we have enough data
if len(X_landmarks) < 100:
    print("\n❌ ERROR: Not enough valid images! Need at least 100 samples.")
    print("   Suggestions:")
    print("   1. Make sure your images clearly show hands")
    print("   2. Ensure good lighting and contrast")
    print("   3. Check that hands aren't too close to camera edges")
    exit()

# ====================== STEP 4: PREPARE DATA FOR TRAINING ======================
print("\n📊 Step 4: Preparing Data for Training...")
print("-"*40)

# Convert to numpy arrays
X = np.array(X_landmarks)
y = np.array(y_labels)

# Encode labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"📊 Data shape: {X.shape}")
print(f"📊 Number of classes: {num_classes}")
print(f"📊 Classes: {list(label_encoder.classes_)}")

# Check class distribution
class_counts = Counter(y)
print("\n📊 Class Distribution:")
for class_name, count in class_counts.most_common(5):
    print(f"   {class_name}: {count} images")
if len(class_counts) > 5:
    print(f"   ... and {len(class_counts)-5} more classes")

# Split data (stratified to maintain class balance)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=TEST_SPLIT, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n📊 Split Statistics:")
print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Testing: {len(X_test)} samples")

# ====================== STEP 5: BUILD THE MODEL ======================
print("\n🏗️ Step 5: Building Neural Network...")
print("-"*40)

def create_landmark_model(input_shape=63, num_classes=num_classes):
    """
    Create a neural network optimized for hand landmark classification
    """
    model = keras.Sequential([
        # Input layer (63 landmarks: 21 points × x,y,z)
        layers.Input(shape=(input_shape,)),
        
        # Reshape to 2D for CNN (7x9 grid - approximation of hand structure)
        layers.Reshape((7, 9, 1)),
        
        # Convolutional layers to learn spatial patterns
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        # Flatten and add dense layers
        layers.Flatten(),
        layers.Dropout(0.3),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create the model
model = create_landmark_model()

# Compile with appropriate optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====================== STEP 6: TRAIN THE MODEL ======================
print("\n🔥 Step 6: Training Model...")
print("-"*40)

# Callbacks for better training
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

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ====================== STEP 7: EVALUATE MODEL ======================
print("\n📈 Step 7: Evaluating Model...")
print("-"*40)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# Get final training metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\n📊 Final Metrics:")
print(f"   Training Accuracy: {final_train_acc*100:.2f}%")
print(f"   Validation Accuracy: {final_val_acc*100:.2f}%")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

# ====================== STEP 8: SAVE MODEL FOR FLUTTER APP ======================
print("\n💾 Step 8: Saving Model for Flutter App...")
print("-"*40)

# Create export directory
export_dir = r"C:\Users\asifa\OneDrive\Desktop\Model\PSL_Exported_Model"
os.makedirs(export_dir, exist_ok=True)

# Save the full Keras model
keras_model_path = os.path.join(export_dir, 'psl_model.h5')
model.save(keras_model_path)
print(f"✅ Saved Keras model: {keras_model_path}")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Reduce model size

try:
    tflite_model = converter.convert()
    tflite_path = os.path.join(export_dir, 'psl_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Saved TFLite model: {tflite_path}")
    
    # Calculate model sizes
    import os
    keras_size = os.path.getsize(keras_model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"   Keras model size: {keras_size:.2f} MB")
    print(f"   TFLite model size: {tflite_size:.2f} MB")
    
except Exception as e:
    print(f"⚠️ TFLite conversion warning: {e}")
    print("   Using standard Keras model instead")

# Save class labels
labels_path = os.path.join(export_dir, 'labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    for label in label_encoder.classes_:
        f.write(f"{label}\n")
print(f"✅ Saved class labels: {labels_path}")

# Save class mapping (for reference)
mapping_path = os.path.join(export_dir, 'class_mapping.txt')
with open(mapping_path, 'w', encoding='utf-8') as f:
    f.write("PSL Class Index Mapping\n")
    f.write("="*30 + "\n")
    for i, label in enumerate(label_encoder.classes_):
        f.write(f"{i}: {label}\n")

# ====================== STEP 9: VISUALIZE RESULTS ======================
print("\n📊 Step 9: Generating Training Visualizations...")
print("-"*40)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(export_dir, 'training_history.png'), dpi=100)
plt.show()

print(f"✅ Saved training plot: {os.path.join(export_dir, 'training_history.png')}")

# ====================== STEP 10: CREATE TEST SCRIPT ======================
print("\n🧪 Step 10: Creating Test Script...")
print("-"*40)

test_script = '''
"""
PSL Model Test Script
Use this to test your trained model on new images
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load model and labels
model = tf.keras.models.load_model('psl_model.h5')
with open('labels.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def predict_sign(image_path):
    # Extract landmarks
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return "No hand detected"
    
    # Get landmarks
    landmarks = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    # Predict
    landmarks_array = np.array(landmarks).reshape(1, -1)
    prediction = model.predict(landmarks_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return labels[predicted_class], confidence

# Test on a sample image
test_image_path = input("Enter path to test image: ")
sign, confidence = predict_sign(test_image_path)
print(f"Predicted Sign: {sign}")
print(f"Confidence: {confidence*100:.2f}%")
'''

test_script_path = os.path.join(export_dir, 'test_model.py')
with open(test_script_path, 'w', encoding='utf-8') as f:
    f.write(test_script)
print(f"✅ Created test script: {test_script_path}")

# ====================== FINAL SUMMARY ======================
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE! 🎉")
print("="*60)
print(f"\n✅ Model saved to: {export_dir}")
print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Validation Accuracy: {final_val_acc*100:.2f}%")

print(f"\n📁 Files created:")
print(f"   1. psl_model.h5 - Full Keras model")
print(f"   2. psl_model.tflite - Optimized for Flutter app")
print(f"   3. labels.txt - Class names for your app")
print(f"   4. class_mapping.txt - Index to class mapping")
print(f"   5. training_history.png - Accuracy/loss graphs")
print(f"   6. test_model.py - Script to test predictions")

print(f"\n🚀 Next Steps for Flutter Integration:")
print(f"   1. Copy 'psl_model.tflite' to your Flutter app's assets folder")
print(f"   2. Copy 'labels.txt' to your Flutter app's assets folder")
print(f"   3. Use tflite_flutter package in your Flutter app")
print(f"   4. Implement real-time hand detection using camera plugin")
print(f"   5. Extract landmarks using MediaPipe in Flutter")
print(f"   6. Feed landmarks to your TFLite model for prediction")

print(f"\n💡 Tips for Better Accuracy:")
print(f"   • Ensure good lighting when capturing signs")
print(f"   • Keep hand centered in frame")
print(f"   • Use consistent background")
print(f"   • Collect more diverse samples if accuracy is low")

print("\n✅ All done! Your PSL recognition system is ready!")