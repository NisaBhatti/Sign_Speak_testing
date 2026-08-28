# Save as: MODEL/model_training/copy_images_simple.py

import os
import shutil
import cv2
import numpy as np

# Source and destination
source_base = r"C:\Users\asifa\OneDrive\Desktop\Model\PSL"
dest_base = r"C:\Users\asifa\OneDrive\Desktop\Model\Simple_Dataset"

# Mapping of Urdu folders to English names (as shown in your folder list)
folder_mapping = {
    'Ain ع': 'Ain',
    'Alif ا': 'Alif',
    'aRay ڑ': 'ArRay',
    'Bari yeh ے': 'BariYeh',
    'Bay ب': 'Bay',
    'Chay چ': 'Chay',
    'Chhoti yeh ی': 'ChhotiYeh',
    'Daal د': 'Daal',
    'Daal ڈ': 'Daal_Heavy',
    'Dhaal ذ': 'Dhaal',
    'Dhuaad ض': 'Dhuaad',
    'Djay ژ': 'Djay',
    'Fay ف': 'Fay',
    'Gaaf گ': 'Gaaf',
    'Ghain غ': 'Ghain',
    'Hamza ‍‌ء': 'Hamza',
    'Hay ہ': 'Hay',
    'hey ح': 'Hey',
    'Jeem ج': 'Jeem',
    'Kaaf ک': 'Kaaf',
    'Khay خ': 'Khay',
    'Laam ل': 'Laam',
    'Meem م': 'Meem',
    'Noon ن': 'Noon',
    'Pay پ': 'Pay',
    'Quaaf ق': 'Quaaf',
    'Ray ر': 'Ray',
    'Seen س': 'Seen',
    'Sheen ‎‎ش': 'Sheen',
    'Suaad ص': 'Suaad',
    'Tay ت': 'Tay',
    'Tey ٹ': 'Tey',
    'Thay ث': 'Thay',
    'Toay\'n ط': 'Toayn',
    'Vao و': 'Vao',
    'Zay ز': 'Zay',
    'Zoay\'n ظ': 'Zoayn',
}

os.makedirs(dest_base, exist_ok=True)

print("📁 Copying images to simple folder names...")
print("="*50)

total_images = 0

for urdu_folder, english_name in folder_mapping.items():
    source_path = os.path.join(source_base, urdu_folder)
    
    if not os.path.exists(source_path):
        print(f"❌ Not found: {urdu_folder}")
        continue
    
    # Create destination folder
    dest_path = os.path.join(dest_base, english_name)
    os.makedirs(dest_path, exist_ok=True)
    
    # Copy all PNG files
    count = 0
    for file in os.listdir(source_path):
        if file.endswith('.png'):
            src_file = os.path.join(source_path, file)
            dst_file = os.path.join(dest_path, file)
            shutil.copy2(src_file, dst_file)
            count += 1
            total_images += 1
    
    print(f"✅ {english_name}: {count} images")

print("="*50)
print(f"\n🎉 Total images copied: {total_images}")
print(f"📁 Destination: {dest_base}")