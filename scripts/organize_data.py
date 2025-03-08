import os
import shutil
from pathlib import Path
import random

def organize_dataset():
    # Create directories if they don't exist
    data_dir = Path("data")
    images_dir = data_dir / "images"
    train_dir = images_dir / "train" / "images"
    val_dir = images_dir / "val" / "images"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    jpeg_files = list(images_dir.glob("*.jpeg"))
    tiff_files = list(images_dir.glob("*.tiff"))
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split ratio (80% train, 20% val)
    train_ratio = 0.8
    
    # Process JPEG files
    random.shuffle(jpeg_files)
    split_idx = int(len(jpeg_files) * train_ratio)
    
    # Move JPEG files
    for jpeg in jpeg_files[:split_idx]:
        shutil.copy2(jpeg, train_dir / jpeg.name)
        print(f"Copied {jpeg.name} to train set")
    
    for jpeg in jpeg_files[split_idx:]:
        shutil.copy2(jpeg, val_dir / jpeg.name)
        print(f"Copied {jpeg.name} to validation set")
    
    # Process TIFF files
    random.shuffle(tiff_files)
    split_idx = int(len(tiff_files) * train_ratio)
    
    # Move TIFF files
    for tiff in tiff_files[:split_idx]:
        shutil.copy2(tiff, train_dir / tiff.name)
        print(f"Copied {tiff.name} to train set")
    
    for tiff in tiff_files[split_idx:]:
        shutil.copy2(tiff, val_dir / tiff.name)
        print(f"Copied {tiff.name} to validation set")
    
    print("\nDataset organization complete!")
    print(f"Train images: {len(list(train_dir.glob('*')))}")
    print(f"Validation images: {len(list(val_dir.glob('*')))}")

if __name__ == "__main__":
    organize_dataset() 