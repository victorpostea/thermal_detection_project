from ultralytics import YOLO
import os
import logging
import cv2
import numpy as np
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def preprocess_thermal_images(data_yaml_path):
    """
    Preprocess all TIFF images in the dataset and save them as JPG for training
    """
    logger = logging.getLogger(__name__)
    logger.info("Preprocessing thermal images...")

    # First, read existing data.yaml to preserve class names
    existing_names = {}
    try:
        with open(data_yaml_path, 'r') as f:
            lines = f.readlines()
            reading_names = False
            for line in lines:
                if line.strip() == "names:":
                    reading_names = True
                elif reading_names and line.strip().startswith("nc:"):
                    reading_names = False
                elif reading_names and ":" in line:
                    idx, name = line.strip().split(":", 1)
                    existing_names[int(idx)] = name.strip()
    except Exception as e:
        logger.warning(f"Could not read existing class names: {e}")
        # Default names if we can't read the existing ones
        existing_names = {
            0: "person",
            1: "bicycle/bike",
            2: "vehicle/car"
        }

    # Create processed directories if they don't exist
    processed_train = Path("data/processed/train")
    processed_val = Path("data/processed/val")
    processed_train.mkdir(exist_ok=True, parents=True)
    processed_val.mkdir(exist_ok=True, parents=True)

    # Create image directories
    processed_train_images = processed_train / "images"
    processed_val_images = processed_val / "images"
    processed_train_images.mkdir(exist_ok=True, parents=True)
    processed_val_images.mkdir(exist_ok=True, parents=True)

    # Process training images
    train_dir = Path("data/images/train/images")
    for tiff_file in train_dir.glob("*.tiff"):
        # Process image
        img = cv2.imread(str(tiff_file), cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint16:
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img_normalized.astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        jpg_path = processed_train_images / f"{tiff_file.stem}.jpg"
        cv2.imwrite(str(jpg_path), img)
        logger.info(f"Processed {tiff_file.name} -> {jpg_path}")

    # Process validation images
    val_dir = Path("data/images/val/images")
    for tiff_file in val_dir.glob("*.tiff"):
        # Process image
        img = cv2.imread(str(tiff_file), cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint16:
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img_normalized.astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        jpg_path = processed_val_images / f"{tiff_file.stem}.jpg"
        cv2.imwrite(str(jpg_path), img)
        logger.info(f"Processed {tiff_file.name} -> {jpg_path}")

    # Update data.yaml with absolute paths while preserving class names
    project_root = Path.cwd()
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {str(project_root)}\n")
        f.write(f"train: {str(processed_train)}\n")
        f.write(f"val: {str(processed_val)}\n")
        f.write("\n")
        f.write("names:\n")
        for idx in sorted(existing_names.keys()):
            f.write(f"  {idx}: {existing_names[idx]}\n")
        f.write("\n")
        f.write("nc: 3  # number of classes\n")

    logger.info("Preprocessing complete!")
    logger.info(f"Updated {data_yaml_path} with processed image paths and preserved class names")

def train_model():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # First, convert the labels
    logger.info("Converting labels to YOLO format...")
    import sys
    import subprocess
    subprocess.run([sys.executable, "scripts/convert_to_yolo.py"], check=True)

    # Then preprocess thermal images
    logger.info("Starting thermal image preprocessing...")
    preprocess_thermal_images('data/data.yaml')

    # Initialize model
    logger.info("Initializing model...")
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Training arguments
    training_args = {
        'data': 'data/data.yaml',        
        'epochs': 200,                   
        'imgsz': 640,                    
        'batch': 1,                      
        'patience': 50,                  
        'device': 'cpu',                 
        'workers': 0,                    
        'project': 'runs/train',         
        'name': 'thermal_yolo',          
        'exist_ok': True,               
        'pretrained': True,             
        'optimizer': 'SGD',             
        'verbose': True,                
        'seed': 42,                     
        'deterministic': True,          
        'single_cls': False,            
        'rect': False,                  
        'cos_lr': True,                
        'close_mosaic': 10,            
        'resume': False,               
        'amp': False,
        'fraction': 1.0,               
        'cache': False,               
        'lr0': 0.01,                  
        'lrf': 0.001,                
        'momentum': 0.937,            
        'weight_decay': 0.0005,       
        'warmup_epochs': 5.0,         
        'warmup_momentum': 0.8,       
        'warmup_bias_lr': 0.1,        
        'box': 7.5,                   
        'cls': 1.5,                   # Increased classification loss weight for better class detection
        'dfl': 1.5,                   
        'plots': False,               
        'save_period': -1,            
        'degrees': 0.0,               
        'translate': 0.2,             
        'scale': 0.5,                 
        'shear': 0.0,                 
        'perspective': 0.0,           
        'flipud': 0.5,                
        'fliplr': 0.5,                
        'mosaic': 1.0,                
        'mixup': 0.1,                 
        'copy_paste': 0.1,            
        'conf': 0.25,                # Confidence threshold for detection
        'iou': 0.5,                  # IoU threshold for NMS
        'task': 'detect',            # Specify task as detection
        'mode': 'train'              # Specify mode as training
    }

    # Start training
    logger.info("Starting training with the following configuration:")
    for key, value in training_args.items():
        logger.info(f"{key}: {value}")

    try:
        results = model.train(**training_args)
        
        # Log training results
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to {results}")
        
        # Validate the model
        logger.info("Running validation...")
        metrics = model.val()
        logger.info("Validation metrics:")
        logger.info(metrics)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
