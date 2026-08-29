# hand_detection_train.py
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class HandDetectionModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """Build a CNN model for hand detection"""
        inputs = keras.Input(shape=self.input_shape)
        
        # First Convolutional Block
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second Convolutional Block
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third Convolutional Block
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Fourth Convolutional Block
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Fifth Convolutional Block
        x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers for classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer (binary classification: hand or no hand)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def build_mobilenet_model(self):
        """Build model using MobileNetV2 as base for better performance"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=base_model.input, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model

class HandDataGenerator:
    def __init__(self, data_dir, image_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        
    def create_data_generators(self):
        """Create data generators for training and validation"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator

class HandDetectionTrainer:
    def __init__(self, model, train_generator, validation_generator):
        self.model = model
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.history = None
        
    def train(self, epochs=50):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_hand_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.validation_generator.batch_size,
            callbacks=callbacks
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def convert_to_tflite(model_path, output_path='hand_detection_model.tflite'):
    """Convert the model to TensorFlow Lite format"""
    # Load the model
    model = keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    
    # For quantization (optional but recommended)
    # You can add representative dataset for quantization
    # converter.representative_dataset = representative_dataset_gen
    
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    
    # Also save as a label map
    labels = {'0': 'no_hand', '1': 'hand'}
    with open('labels.json', 'w') as f:
        json.dump(labels, f)

def collect_data_from_camera(num_samples=500, save_dir='hand_dataset'):
    """Collect data from camera for training"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create directories
    for category in ['hand', 'no_hand']:
        os.makedirs(os.path.join(save_dir, category), exist_ok=True)
    
    hand_count = 0
    no_hand_count = 0
    
    print("Collecting data...")
    print("Press 'h' to capture hand image")
    print("Press 'n' to capture no-hand image")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        display = frame.copy()
        cv2.putText(display, f"Hand: {hand_count}, No-Hand: {no_hand_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collection', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('h') and hand_count < num_samples:
            img_name = os.path.join(save_dir, 'hand', f'hand_{hand_count}.jpg')
            cv2.imwrite(img_name, frame)
            hand_count += 1
            print(f"Saved hand image {hand_count}/{num_samples}")
            
        elif key == ord('n') and no_hand_count < num_samples:
            img_name = os.path.join(save_dir, 'no_hand', f'no_hand_{no_hand_count}.jpg')
            cv2.imwrite(img_name, frame)
            no_hand_count += 1
            print(f"Saved no-hand image {no_hand_count}/{num_samples}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete! Hand: {hand_count}, No-Hand: {no_hand_count}")

def create_synthetic_data():
    """Create synthetic hand images using data augmentation"""
    # This is a placeholder for creating synthetic data
    # You can use techniques like:
    # 1. Image augmentation from existing hand images
    # 2. Hand skeleton generation
    # 3. Texture mapping on hand templates
    pass

def download_hand_dataset():
    """Download hand detection datasets from the internet"""
    # You can download datasets like:
    # 1. EgoHands Dataset
    # 2. Oxford Hands Dataset
    # 3. CMU Hand Dataset
    
    import subprocess
    import sys
    
    def install_package(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    try:
        import kagglehub
    except ImportError:
        install_package("kagglehub")
        import kagglehub
    
    # Download a sample hand dataset (you'll need to find appropriate dataset)
    print("Downloading hand dataset...")
    # Example: kagglehub.dataset_download("username/dataset_name")
    print("Please manually download hand dataset and organize in directories")

# Main training script
def main():
    # Configuration
    DATA_DIR = 'hand_dataset'  # Should contain 'hand' and 'no_hand' subdirectories
    EPOCHS = 30
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' not found!")
        print("Options:")
        print("1. Collect data from camera using collect_data_from_camera()")
        print("2. Download a dataset using download_hand_dataset()")
        print("3. Manually create dataset with 'hand' and 'no_hand' directories")
        
        choice = input("Choose option (1/2/3): ")
        if choice == '1':
            collect_data_from_camera()
        elif choice == '2':
            download_hand_dataset()
        else:
            print("Please create dataset directory manually and rerun")
            return
    
    print("Creating data generators...")
    data_gen = HandDataGenerator(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    train_gen, val_gen = data_gen.create_data_generators()
    
    print("Building model...")
    model = HandDetectionModel(IMAGE_SIZE + (3,))
    # Choose model type
    use_mobilenet = input("Use MobileNetV2? (y/n): ").lower() == 'y'
    
    if use_mobilenet:
        model.build_mobilenet_model()
    else:
        model.build_model()
    
    print("Model summary:")
    model.model.summary()
    
    print("Training model...")
    trainer = HandDetectionTrainer(model.model, train_gen, val_gen)
    history = trainer.train(epochs=EPOCHS)
    
    print("Plotting training history...")
    trainer.plot_training_history()
    
    print("Saving model...")
    model.model.save('hand_detection_model.h5')
    
    print("Converting to TFLite...")
    convert_to_tflite('hand_detection_model.h5', 'hand_detection_model.tflite')
    
    print("Training complete!")

if __name__ == "__main__":
    main()