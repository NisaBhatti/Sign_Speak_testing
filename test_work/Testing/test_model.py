# huggingface_hand_models.py
"""
Download pre-trained hand models from Hugging Face
"""

import os
from huggingface_hub import hf_hub_download

def download_from_huggingface():
    """Download pre-trained hand detection models from Hugging Face"""
    
    print("\n📥 Downloading from Hugging Face...")
    
    os.makedirs('assets/models', exist_ok=True)
    
    # Available hand detection models on Hugging Face
    models = {
        # Hand landmark detection
        "hand_landmark": {
            "repo": "vinid/plab",
            "filename": "hand_landmark_model.tflite"
        },
        # YOLO hand detection
        "yolo_hand": {
            "repo": "hologerry/yolov8n-hand",
            "filename": "yolov8n-hand.pt"
        },
        # MediaPipe style model
        "mediapipe_hand": {
            "repo": "google/mediapipe",
            "filename": "hand_landmarker.task"
        }
    }
    
    downloaded = []
    
    for name, model_info in models.items():
        try:
            print(f"Downloading {name}...")
            filepath = hf_hub_download(
                repo_id=model_info["repo"],
                filename=model_info["filename"],
                local_dir="assets/models",
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded: {filepath}")
            downloaded.append(filepath)
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
    
    return downloaded

if __name__ == "__main__":
    download_from_huggingface()