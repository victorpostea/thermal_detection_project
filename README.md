# Thermal Object Detection Project

This project implements a YOLOv8-based object detection system for thermal images, capable of detecting people, bicycles/bikes, and vehicles/cars in thermal imagery.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Processing

1. Place your thermal images in the following structure:
```
data/
├── images/
│   ├── train/
│   │   └── images/  # Training TIFF images
│   └── val/
│       └── images/  # Validation TIFF images
└── labels_test_16_bit.txt  # Original labels file
```

2. Process the data:
```bash
python scripts/convert_to_yolo.py  # Convert labels to YOLO format
```

The script will:
- Convert 16-bit TIFF images to 8-bit JPG
- Apply INFERNO colormap for better visualization
- Convert labels to YOLO format
- Create processed dataset in `data/processed/`

## Training

Train the model with optimized settings for thermal detection:
```bash
python scripts/train.py
```

The training script:
- Uses YOLOv8 nano model
- Runs for 200 epochs
- Applies class weighting for better bicycle detection
- Uses multi-scale training
- Saves best model to `runs/train/thermal_yolo/weights/best.pt`

## Evaluation

Evaluate model performance on the validation set:
```bash
python scripts/evaluate.py
```

This will:
- Show images with ground truth (green) and predictions (blue)
- Calculate precision, recall, and F1-score for each class
- Display IoU (Intersection over Union) metrics
- Press any key to move to next image, 'q' to quit

## Inference

Run inference on a single image or directory:
```bash
python scripts/inference.py --source path/to/image_or_dir
```

## Live Detection

Run real-time detection on thermal camera feed:
```bash
python app.py
```

The app will:
- Connect to your thermal camera
- Process frames in real-time
- Display detections with confidence scores
- Press 'q' to quit

## Model Performance

The model is optimized for thermal imagery with:
- Class weights to handle imbalanced data
- Multi-scale training for better small object detection
- Augmentation techniques specific to thermal images

## Classes

The model detects three classes:
1. Person
2. Bicycle/Bike
3. Vehicle/Car

## Troubleshooting

- If detection confidence is low for bicycles, adjust the confidence threshold:
  ```python
  model.predict(source=image_path, conf=0.25)  # Lower threshold for more detections
  ```
- For memory issues, reduce batch size in training_args
- For GPU support, set device='0' in training_args
