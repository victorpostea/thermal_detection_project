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

    # Update data.yaml with absolute paths
    project_root = Path.cwd()
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {str(project_root)}\n")
        f.write(f"train: {str(processed_train)}\n")
        f.write(f"val: {str(processed_val)}\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: Pedestrian\n")
        f.write("  1: Cyclist\n")
        f.write("  2: Car\n")
        f.write("\n")
        f.write("nc: 3  # number of classes\n")

    logger.info("Preprocessing complete!")
    logger.info(f"Updated {data_yaml_path} with processed image paths")

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
        'data': 'data/data.yaml',        # path to data.yaml
        'epochs': 200,                   # number of epochs (increased from 50)
        'imgsz': 640,                    # image size (increased from 320)
        'batch': 1,                      # batch size
        'patience': 50,                  # early stopping patience (increased from 20)
        'device': 'cpu',                 # device to use
        'workers': 0,                    # number of worker threads
        'project': 'runs/train',         # project name
        'name': 'thermal_yolo',          # experiment name
        'exist_ok': True,               # overwrite existing experiment
        'pretrained': True,             # use pretrained model
        'optimizer': 'SGD',             # optimizer
        'verbose': True,                # verbose output
        'seed': 42,                     # random seed
        'deterministic': True,          # deterministic training
        'single_cls': False,            # train as single-class dataset
        'rect': False,                  # rectangular training
        'cos_lr': True,                # cosine learning rate scheduler
        'close_mosaic': 10,            # disable mosaic augmentation for final epochs
        'resume': False,               # resume training from last checkpoint
        'amp': False,                  # Automatic Mixed Precision (disabled)
        'fraction': 1.0,               # dataset fraction to train on
        'cache': False,               # cache images for faster training
        'lr0': 0.01,                  # initial learning rate
        'lrf': 0.001,                 # final learning rate (decreased from 0.01)
        'momentum': 0.937,            # SGD momentum/Adam beta1
        'weight_decay': 0.0005,       # optimizer weight decay
        'warmup_epochs': 5.0,         # warmup epochs (increased from 3.0)
        'warmup_momentum': 0.8,       # warmup initial momentum
        'warmup_bias_lr': 0.1,        # warmup initial bias lr
        'box': 7.5,                   # box loss gain
        'cls': 0.5,                   # cls loss gain
        'dfl': 1.5,                   # dfl loss gain
        'plots': False,               # disable plotting (saves memory)
        'save_period': -1,            # save checkpoint every x epochs (-1 to disable)
        'degrees': 0.0,               # rotation augmentation
        'translate': 0.2,             # translation augmentation (increased from 0.1)
        'scale': 0.5,                 # scale augmentation
        'shear': 0.0,                 # shear augmentation
        'perspective': 0.0,           # perspective augmentation
        'flipud': 0.5,                # vertical flip augmentation (added)
        'fliplr': 0.5,                # horizontal flip augmentation
        'mosaic': 1.0,                # mosaic augmentation
        'mixup': 0.1,                 # mixup augmentation (added)
        'copy_paste': 0.1,            # copy-paste augmentation (added)
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
