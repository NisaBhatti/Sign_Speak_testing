# Save as: MODEL/model_training/train_simple_folders.py

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

print("🚀 Training Urdu Alphabet Recognition")
print("="*50)

# Use the simple dataset we just created
simple_dataset_path = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset"

# Get all alphabet folders
alphabets = [f for f in os.listdir(simple_dataset_path) 
             if os.path.isdir(os.path.join(simple_dataset_path, f))]

print(f"📊 Found {len(alphabets)} alphabets")
print(f"Alphabets: {alphabets[:10]}...")  # Show first 10

# Load all images
X = []  # Images
y = []  # Labels

for label_idx, alphabet in enumerate(alphabets):
    folder_path = os.path.join(simple_dataset_path, alphabet)
    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(label_idx)
            print(f"✅ Loaded: {alphabet}/{img_file}")
    
    print(f"📁 {alphabet}: {len(images)} images")

X = np.array(X)
y = np.array(y)

print(f"\n📊 Total images: {len(X)}")
print(f"📊 Total classes: {len(alphabets)}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Training: {len(X_train)} images")
print(f"📊 Validation: {len(X_val)} images")

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(alphabets))
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(alphabets))

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    horizontal_flip=False
)

# Normalize validation data
X_val = X_val / 255.0

# MobileNetV2 Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(alphabets), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
print("\n🔥 Training started...")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val_cat),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ],
    verbose=1
)

# Save model
export_path = r"C:\Users\asifa\OneDrive\Desktop\Model\Exported_Model"
os.makedirs(export_path, exist_ok=True)

# Save Keras model
model.save(os.path.join(export_path, 'urdu_alphabets_full.h5'))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(os.path.join(export_path, 'urdu_alphabets_full.tflite'), 'wb') as f:
    f.write(tflite_model)

# Save labels
with open(os.path.join(export_path, 'labels.txt'), 'w', encoding='utf-8') as f:
    for alphabet in alphabets:
        f.write(f"{alphabet}\n")

print(f"\n✅ Model saved to: {export_path}")
print(f"✅ Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"✅ Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(os.path.join(export_path, 'training_history.png'))
plt.show()