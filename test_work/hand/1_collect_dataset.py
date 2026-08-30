# 0_setup_and_download.py
"""
Complete setup: Install dependencies and download dataset
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "tensorflow",
        "opencv-python",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "kagglehub",
        "pillow"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All dependencies installed!")

def download_dataset():
    """Download the hand detection dataset"""
    print("\n📥 Downloading Hand Detection Dataset...")
    
    try:
        import kagglehub
        
        # Download dataset
        path = kagglehub.dataset_download("nomihsa965/hand-detection-dataset-vocyolo-format")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except ImportError:
        print("❌ kagglehub not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        
        import kagglehub
        path = kagglehub.dataset_download("nomihsa965/hand-detection-dataset-vocyolo-format")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n💡 Manual download option:")
        print("1. Visit: https://www.kaggle.com/datasets/nomihsa965/hand-detection-dataset-vocyolo-format")
        print("2. Click Download")
        print("3. Extract to 'hand_dataset' folder")
        return None

def main():
    print("=" * 70)
    print("   HAND DETECTION SETUP")
    print("=" * 70)
    
    # Install dependencies
    install_dependencies()
    
    # Download dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        print(f"\n✅ Setup complete!")
        print(f"   Dataset path: {dataset_path}")
        print(f"   Next: python 1_prepare_dataset.py")
    else:
        print("\n⚠️ Please download dataset manually and run 1_prepare_dataset.py")

if __name__ == "__main__":
    main()