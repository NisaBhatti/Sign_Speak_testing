# download_pretrained_hand_model.py
"""
Simple script to download pre-trained hand detection models
No HuggingFace required - uses direct downloads
"""

import os
import urllib.request
import json
import zipfile
import subprocess
import sys

def install_package(package):
    """Install a Python package"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_file(url, filename):
    """Download a file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        print(f"\rDownloading: {percent}%", end="")
    
    print(f"\n📥 Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def download_mediapipe_model():
    """Download MediaPipe hand landmark model (Best option)"""
    print("\n" + "="*60)
    print("   DOWNLOADING MEDIAPIPE HAND MODEL")
    print("="*60)
    
    os.makedirs('assets/models', exist_ok=True)
    
    # MediaPipe hand landmark model
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    output = "assets/models/hand_landmarker.task"
    
    if download_file(url, output):
        print(f"✅ Model saved: {output}")
        return output
    return None

def download_tflite_hand_models():
    """Download TFLite hand detection models"""
    print("\n" + "="*60)
    print("   DOWNLOADING TFLITE HAND MODELS")
    print("="*60)
    
    os.makedirs('assets/models', exist_ok=True)
    
    # Alternative TFLite models
    models = {
        "hand_landmarker.tflite": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.tflite",
        "palm_detection.tflite": "https://storage.googleapis.com/mediapipe-models/palm_detection/palm_detection/float16/1/palm_detection.tflite"
    }
    
    downloaded = []
    for filename, url in models.items():
        output = f"assets/models/{filename}"
        if download_file(url, output):
            downloaded.append(output)
    
    return downloaded

def download_sample_hand_dataset():
    """Download a sample hand dataset for testing"""
    print("\n" + "="*60)
    print("   DOWNLOADING SAMPLE HAND DATASET")
    print("="*60)
    
    # This is a small sample dataset
    # You can replace this with any hand dataset URL
    dataset_urls = [
        "https://www.kaggle.com/datasets/divyanshrai/hand-gesture-recognition-dataset",
        "https://www.kaggle.com/datasets/atulanandjha/hand-gesture-recognition-dataset",
    ]
    
    print("\n📊 Sample hand datasets can be downloaded from:")
    for url in dataset_urls:
        print(f"  - {url}")
    
    print("\n💡 For quick testing, use option 1 (MediaPipe) which needs no dataset")

def create_metadata_file():
    """Create metadata for the models"""
    metadata = {
        "model_name": "Hand Landmark Detection",
        "description": "Pre-trained MediaPipe hand landmark model",
        "num_landmarks": 21,
        "input_size": [224, 224, 3],
        "landmarks": [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ],
        "connections": [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
        ]
    }
    
    with open('assets/models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Created: assets/models/metadata.json")

def create_test_script():
    """Create a test script for the downloaded model"""
    test_script = '''# test_hand_model.py
"""
Test the downloaded hand detection model
"""

import cv2
import numpy as np
import os

def test_mediapipe():
    """Test MediaPipe hand detection"""
    try:
        import mediapipe as mp
        
        print("📷 Testing MediaPipe Hand Detection...")
        print("Press 'q' to quit")
        
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not found!")
            return
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue
                
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Show number of landmarks
                        cv2.putText(image, f"Landmarks: {len(hand_landmarks.landmark)}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "No hand detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Hand Detection', image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except ImportError:
        print("❌ MediaPipe not installed!")
        print("Install with: pip install mediapipe")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_tflite():
    """Test TFLite model"""
    try:
        import tensorflow as tf
        
        print("🧪 Testing TFLite model...")
        
        model_path = "assets/models/hand_landmarker.tflite"
        if not os.path.exists(model_path):
            print("❌ Model file not found!")
            return
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("Install with: pip install tensorflow")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("   HAND MODEL TEST")
    print("="*60)
    print("\\n1. Test MediaPipe Hand Detection")
    print("2. Test TFLite Model")
    print("3. Both")
    
    choice = input("\\nEnter choice (1-3): ")
    
    if choice == '1':
        test_mediapipe()
    elif choice == '2':
        test_tflite()
    elif choice == '3':
        test_tflite()
        test_mediapipe()
    else:
        print("Invalid choice")
'''
    
    with open('test_hand_model.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created: test_hand_model.py")

def main():
    print("="*70)
    print("   PRE-TRAINED HAND MODEL DOWNLOADER")
    print("="*70)
    
    print("\n📌 Select an option:")
    print("1. Download MediaPipe Hand Landmark Model (Recommended) ⭐")
    print("2. Download TFLite Hand Models")
    print("3. Download both models")
    print("4. Show dataset options")
    print("5. Install dependencies")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == '1':
        model = download_mediapipe_model()
        if model:
            create_metadata_file()
            create_test_script()
            print("\n✅ Setup complete! Run: python test_hand_model.py")
    
    elif choice == '2':
        models = download_tflite_hand_models()
        if models:
            create_metadata_file()
            create_test_script()
            print("\n✅ Setup complete! Run: python test_hand_model.py")
    
    elif choice == '3':
        download_mediapipe_model()
        download_tflite_hand_models()
        create_metadata_file()
        create_test_script()
        print("\n✅ Setup complete! Run: python test_hand_model.py")
    
    elif choice == '4':
        download_sample_hand_dataset()
    
    elif choice == '5':
        print("\n📦 Installing dependencies...")
        packages = ["mediapipe", "opencv-python", "tensorflow"]
        for package in packages:
            try:
                install_package(package)
                print(f"✅ Installed: {package}")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")
    
    elif choice == '6':
        print("Goodbye!")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()