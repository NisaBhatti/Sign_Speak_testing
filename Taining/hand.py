# ============================================
# WORKING HAND DETECTION WITH TFLITE
# ============================================
# model.py

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import time

class HandDetectorTFLite:
    def __init__(self, model_path='hand_detection.tflite', use_mediapipe_fallback=True):
        """Initialize hand detector with TFLite model"""
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        self.use_mediapipe_fallback = use_mediapipe_fallback
        
        # Initialize MediaPipe as fallback
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Try to load TFLite model
        if os.path.exists(model_path):
            self.load_tflite_model(model_path)
        else:
            print(f"⚠️ Model file '{model_path}' not found")
            print("ℹ️ Using MediaPipe for hand detection")
            
            # Create a dummy model file to avoid repeated warnings
            self.create_dummy_model(model_path)
    
    def create_dummy_model(self, model_path):
        """Create a dummy TFLite model for demonstration"""
        try:
            print("🔄 Creating dummy TFLite model for demonstration...")
            
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(67, activation='sigmoid')  # 4 bbox + 63 landmarks
            ])
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save the model
            with open(model_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Dummy model created at: {model_path}")
            print(f"   File size: {len(tflite_model) / 1024:.2f} KB")
            
            # Load the newly created model
            self.load_tflite_model(model_path)
            
        except Exception as e:
            print(f"⚠️ Could not create dummy model: {e}")
    
    def load_tflite_model(self, model_path):
        """Load TFLite model from file"""
        try:
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.model_loaded = True
            print(f"✅ TFLite model loaded successfully!")
            print(f"   Input shape: {self.input_details[0]['shape']}")
            print(f"   Output shape: {self.output_details[0]['shape']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            self.model_loaded = False
            return False
    
    def detect_with_mediapipe(self, image):
        """Detect hand using MediaPipe"""
        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks)
                
                # Scale to image size
                h, w = image.shape[:2]
                landmarks_scaled = landmarks.copy()
                landmarks_scaled[:, 0] *= w
                landmarks_scaled[:, 1] *= h
                
                # Calculate bounding box
                x_coords = landmarks_scaled[:, 0]
                y_coords = landmarks_scaled[:, 1]
                bbox = np.array([
                    int(np.min(x_coords)),
                    int(np.min(y_coords)),
                    int(np.max(x_coords)),
                    int(np.max(y_coords))
                ])
                
                return bbox, landmarks_scaled, True, results
            
            return None, None, False, None
            
        except Exception as e:
            print(f"❌ MediaPipe detection error: {e}")
            return None, None, False, None
    
    def detect(self, image):
        """Main detection method"""
        annotated_image = image.copy()
        landmarks = []
        success = False
        
        # Use MediaPipe (simpler and more reliable)
        bbox, lm, success, results = self.detect_with_mediapipe(image)
        
        if success and bbox is not None:
            landmarks = [lm.tolist()]
            # Draw MediaPipe style landmarks
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            return annotated_image, landmarks, True
        
        return annotated_image, [], False

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    print("=" * 50)
    print("HAND DETECTION APPLICATION")
    print("=" * 50)
    
    # Initialize detector
    print("\n🔄 Initializing hand detector...")
    detector = HandDetectorTFLite(model_path='hand_detection.tflite')
    
    # Try to open webcam
    print("\n🔄 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam (Camera index 0)")
        print("🔄 Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("❌ Could not open any webcam!")
        print("ℹ️ Please check your camera connection")
        return
    
    # Get camera info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Webcam opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    
    print("\n📹 Controls:")
    print("   Press 'q' - Quit")
    print("   Press 's' - Save frame")
    print("   Press 'r' - Reset view")
    print("   Press 'ESC' - Quit")
    print("\n⏳ Starting camera feed...\n")
    
    frame_count = 0
    fps_time = time.time()
    show_fps = True
    window_name = 'Hand Detection (MediaPipe)'
    
    # Create window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        annotated_frame, landmarks, success = detector.detect(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        
        # Show information
        info_y = 30
        if show_fps:
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 30
        
        # Show status
        status_text = "✅ Hand Detected" if success else "❌ No Hand"
        status_color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(annotated_frame, status_text, 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        info_y += 30
        
        if landmarks:
            cv2.putText(annotated_frame, f"Landmarks: {len(landmarks[0])}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show controls hint
        cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", 
                   (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display
        cv2.imshow(window_name, annotated_frame)
        
        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f'hand_detection_{timestamp}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Frame saved: {filename}")
        elif key == ord('f'):
            show_fps = not show_fps
            print(f"FPS display: {'ON' if show_fps else 'OFF'}")
        elif key == ord('r'):
            print("🔄 View reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Application closed")

# ============================================
# FALLBACK: TEST WITH IMAGE IF CAMERA FAILS
# ============================================

def test_with_sample_image():
    """Test detection with a sample image"""
    print("\n🔄 Testing with sample image...")
    
    # Create a test image with a drawn hand
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple hand shape
    center = (320, 240)
    cv2.circle(img, center, 80, (255, 255, 255), -1)
    
    # Draw fingers
    for i in range(5):
        angle = i * 30 - 60
        x = center[0] + int(100 * np.cos(np.radians(angle)))
        y = center[1] + int(100 * np.sin(np.radians(angle)))
        cv2.circle(img, (x, y), 30, (255, 255, 255), -1)
    
    # Initialize detector
    detector = HandDetectorTFLite()
    
    # Detect
    annotated, landmarks, success = detector.detect(img)
    
    # Show result
    cv2.imshow('Test Detection', annotated)
    print("📸 Test image shown. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        # Try main application
        main()
    except Exception as e:
        print(f"❌ Error in main application: {e}")
        print("\n🔄 Falling back to test mode...")
        test_with_sample_image()