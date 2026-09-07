import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TEY (ط) SIGN DETECTION - EXCLUSIVE TRAINING")
print("="*60)

# ==================== CONFIGURATION ====================
DATASET_PATH = r"D:\MODEL\Sign_Speak_testing-main\Simple_Dataset\Tey"
MODEL_SAVE_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\tey_model_exclusive.tflite"
LABEL_ENCODER_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\tey_label_encoder_exclusive.pkl"
CONFIG_PATH = r"D:\MODEL\Sign_Speak_testing-main\Exported_Model\tey_config_exclusive.json"
HISTORY_PLOT_PATH = "tey_training_history_exclusive.png"

# Create directories
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ==================== MEDIAPIPE SETUP ====================
print("\n🔄 Initializing MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
print("✅ MediaPipe initialized")

# ==================== FEATURE EXTRACTION ====================
def extract_hand_landmarks(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
        
        if len(landmarks) >= 42:
            wrist_x, wrist_y = landmarks[0], landmarks[1]
            for i in range(0, len(landmarks), 2):
                landmarks[i] -= wrist_x
                landmarks[i+1] -= wrist_y
            return np.array(landmarks, dtype=np.float32)
        
        return None
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# ==================== LOAD DATASET ====================
print(f"\n📂 Loading dataset from: {DATASET_PATH}")

# Load Tey images
tey_images = []
print("\n📸 Loading Tey images...")
for img_file in os.listdir(DATASET_PATH):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(DATASET_PATH, img_file)
        landmarks = extract_hand_landmarks(img_path)
        if landmarks is not None:
            tey_images.append(landmarks)
            print(f"  ✓ Tey: {img_file}")

print(f"✅ Loaded {len(tey_images)} Tey images")

# Load non-Tey images
print("\n📸 Loading non-Tey images (ALL OTHER SIGNS)...")
non_tey_images = []
PARENT_PATH = os.path.dirname(DATASET_PATH)
alif_count = 0
other_count = 0

if os.path.exists(PARENT_PATH):
    for root, dirs, files in os.walk(PARENT_PATH):
        if root == DATASET_PATH or any(skip in root for skip in ['__pycache__', '.git', 'Exported_Model']):
            continue
        
        folder_name = os.path.basename(root)
        
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(root, img_file)
                landmarks = extract_hand_landmarks(img_path)
                if landmarks is not None:
                    non_tey_images.append(landmarks)
                    if 'Alif' in folder_name:
                        alif_count += 1
                        print(f"  ✓ ALIF (NOT TEY): {img_file}")
                    else:
                        other_count += 1
                        if other_count % 50 == 0:
                            print(f"  ✓ Other sign: {img_file} (from {folder_name})")

print(f"\n✅ Loaded non-Tey images:")
print(f"   Alif examples: {alif_count}")
print(f"   Other signs: {other_count}")
print(f"   Total non-Tey: {len(non_tey_images)}")

# ==================== PREPARE DATA ====================
print("\n📊 Preparing dataset...")

# Combine data
all_images = tey_images + non_tey_images
all_labels = ['tey'] * len(tey_images) + ['not_tey'] * len(non_tey_images)

X = np.array(all_images, dtype=np.float32)
y = np.array(all_labels)

print(f"\n📊 Dataset Summary:")
print(f"   Tey samples: {len(tey_images)}")
print(f"   Non-Tey samples: {len(non_tey_images)}")
print(f"   Total: {len(X)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\n🔢 Label Encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"   {label} -> {i}")

# ==================== SPLIT DATA ====================
print("\n🔄 Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"\n📊 Train/Test Split:")
print(f"   Training: {len(X_train)}")
print(f"   Test: {len(X_test)}")

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print(f"\n⚖️ Class Weights:")
for class_id, weight in class_weight_dict.items():
    class_name = label_encoder.inverse_transform([class_id])[0]
    print(f"   {class_name}: {weight:.3f}")

# ==================== AGGRESSIVE DATA AUGMENTATION ====================
def augment_landmarks_aggressive(landmarks, label, num_augmentations=5):
    """Apply aggressive data augmentation"""
    augmented = []
    augmented_labels = []
    
    # Original
    augmented.append(landmarks)
    augmented_labels.append(label)
    
    for _ in range(num_augmentations):
        aug_landmarks = landmarks.copy()
        
        # Multiple noise levels
        noise = np.random.normal(0, np.random.uniform(0.01, 0.03), landmarks.shape)
        aug_landmarks = aug_landmarks + noise
        
        # Random scaling
        scale = np.random.uniform(0.85, 1.15)
        aug_landmarks = aug_landmarks * scale
        
        # Random rotation
        angle = np.random.uniform(-0.2, 0.2)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        reshaped = aug_landmarks.reshape(-1, 2)
        rotated = reshaped @ rotation_matrix
        aug_landmarks = rotated.flatten()
        
        # Random shift
        shift = np.random.uniform(-0.08, 0.08, 42)
        aug_landmarks = aug_landmarks + shift
        
        # Random mirror
        if np.random.random() > 0.5:
            for i in range(0, len(aug_landmarks), 2):
                aug_landmarks[i] = -aug_landmarks[i]
        
        augmented.append(aug_landmarks.astype(np.float32))
        augmented_labels.append(label)
    
    return augmented, augmented_labels

print("\n🔄 Applying aggressive augmentation...")

X_train_aug = []
y_train_aug = []

for landmarks, label in zip(X_train, y_train):
    if label == 1:  # Tey - augment more
        aug_landmarks, aug_labels = augment_landmarks_aggressive(landmarks, label, num_augmentations=6)
    else:  # Non-Tey - also augment but less
        aug_landmarks, aug_labels = augment_landmarks_aggressive(landmarks, label, num_augmentations=2)
    
    X_train_aug.extend(aug_landmarks)
    y_train_aug.extend(aug_labels)

X_train_aug = np.array(X_train_aug, dtype=np.float32)
y_train_aug = np.array(y_train_aug)

print(f"✅ Augmentation complete:")
print(f"   Original: {len(X_train)}")
print(f"   Augmented: {len(X_train_aug)}")

# ==================== BUILD EXCLUSIVE MODEL ====================
print("\n🏗️ Building exclusive model...")

def create_exclusive_model():
    model = keras.Sequential([
        layers.Input(shape=(42,)),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = create_exclusive_model()

# Compile with aggressive settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==================== TRAINING ====================
print("\n🚀 Starting training...")
print("   Press Ctrl+C to stop early")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=25,
        min_lr=0.000001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_tey_exclusive.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

try:
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 3.0},  # Give Tey 3x more weight
        verbose=1
    )
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user. Saving current model...")
except Exception as e:
    print(f"\n⚠️ Training error: {e}")
    print("   Attempting to save current model...")

# Load best model if exists
if os.path.exists('best_tey_exclusive.h5'):
    model.load_weights('best_tey_exclusive.h5')
    print("✅ Loaded best model from checkpoint")
else:
    print("⚠️ No checkpoint found. Using current model.")

# ==================== EVALUATION ====================
print("\n📊 Evaluating model...")

try:
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    test_precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy:  {test_accuracy*100:.2f}%")
    print(f"   Precision: {test_precision*100:.2f}%")
    print(f"   Recall:    {test_recall*100:.2f}%")
    print(f"   F1-Score:  {test_f1*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   ┌─────────────────────┬─────────────────────┐")
    print(f"   │                     │    Predicted        │")
    print(f"   │                     ├──────────┬──────────┤")
    print(f"   │                     │ Not Tey  │   Tey    │")
    print(f"   ├─────────────────────┼──────────┼──────────┤")
    print(f"   │ Actual Not Tey      │   {cm[0][0]:>3}     │   {cm[0][1]:>3}     │")
    print(f"   │ Actual Tey          │   {cm[1][0]:>3}     │   {cm[1][1]:>3}     │")
    print(f"   └─────────────────────┴──────────┴──────────┘")
    
    # Test different thresholds
    print(f"\n📊 Testing different confidence thresholds:")
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred_t = (y_pred_proba > thresh).astype(int).flatten()
        acc = np.mean(y_pred_t == y_test)
        prec = precision_score(y_test, y_pred_t, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred_t, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred_t, average='binary', zero_division=0)
        print(f"   Threshold {thresh:.2f}: Acc={acc*100:.1f}% | Prec={prec*100:.1f}% | Rec={rec*100:.1f}% | F1={f1*100:.1f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\n✅ Best threshold: {best_threshold:.2f} (F1: {best_f1*100:.1f}%)")
    
    # Use best threshold for final evaluation
    BEST_THRESHOLD = max(0.7, best_threshold)  # Use at least 0.7 for exclusivity
    y_pred_final = (y_pred_proba > BEST_THRESHOLD).astype(int).flatten()
    
    final_precision = precision_score(y_test, y_pred_final, average='binary', zero_division=0)
    final_recall = recall_score(y_test, y_pred_final, average='binary', zero_division=0)
    final_f1 = f1_score(y_test, y_pred_final, average='binary', zero_division=0)
    final_accuracy = np.mean(y_pred_final == y_test)
    
    print(f"\n📊 Final Results (Threshold: {BEST_THRESHOLD}):")
    print(f"   Accuracy:  {final_accuracy*100:.2f}%")
    print(f"   Precision: {final_precision*100:.2f}%")
    print(f"   Recall:    {final_recall*100:.2f}%")
    print(f"   F1-Score:  {final_f1*100:.2f}%")

except Exception as e:
    print(f"⚠️ Error during evaluation: {e}")
    BEST_THRESHOLD = 0.7
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0

# ==================== SAVE MODEL ====================
print("\n💾 Saving model...")

try:
    # Save as Keras model
    model.save('tey_exclusive_model.h5')
    print("✅ Keras model saved to: tey_exclusive_model.h5")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_model = converter.convert()
        with open(MODEL_SAVE_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model saved to: {MODEL_SAVE_PATH}")
    except:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(MODEL_SAVE_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model (float) saved to: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"⚠️ Error saving model: {e}")

# ==================== SAVE CONFIG ====================
config = {
    "model_path": MODEL_SAVE_PATH,
    "label_encoder_path": LABEL_ENCODER_PATH,
    "threshold": float(BEST_THRESHOLD),
    "metrics": {
        "accuracy": float(test_accuracy) if 'test_accuracy' in locals() else 0,
        "precision": float(test_precision) if 'test_precision' in locals() else 0,
        "recall": float(test_recall) if 'test_recall' in locals() else 0,
        "f1_score": float(test_f1) if 'test_f1' in locals() else 0
    },
    "training_config": {
        "tey_samples": len(tey_images),
        "non_tey_samples": len(non_tey_images),
        "total_samples": len(X),
        "training_samples": len(X_train_aug)
    }
}

with open(CONFIG_PATH, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Config saved to: {CONFIG_PATH}")

# ==================== CLEANUP ====================
hands.close()

print("\n" + "="*60)
print("✅ EXCLUSIVE TEY MODEL TRAINING COMPLETED!")
print("="*60)
print(f"\n📁 Model saved to: {MODEL_SAVE_PATH}")
print(f"📁 Config saved to: {CONFIG_PATH}")
print(f"\n💡 This model is trained to ONLY detect TEY (ط)")
print("   All other signs (including Alif) will be rejected")
print(f"   Confidence threshold: {BEST_THRESHOLD}")
print("="*60)