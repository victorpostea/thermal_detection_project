from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd

def load_ground_truth(label_file):
    """Load original labels from the label file"""
    df = pd.read_csv(
        label_file,
        sep=" ",
        names=["filename", "class_id", "xmin", "ymin", "xmax", "ymax"]
    )
    # Convert .tiff to .jpg in filenames
    df['filename'] = df['filename'].str.replace('.tiff', '.jpg')
    return df

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    # box format: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_model(model_path, image_dir, label_file, iou_threshold=0.5):
    """Evaluate model performance against ground truth"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Load ground truth labels
    gt_labels = load_ground_truth(label_file)
    
    # Initialize metrics
    class_metrics = {
        1: {"name": "person", "TP": 0, "FP": 0, "FN": 0},
        2: {"name": "bicycle/bike", "TP": 0, "FP": 0, "FN": 0},
        3: {"name": "vehicle/car", "TP": 0, "FP": 0, "FN": 0}
    }
    
    # Process each image
    for filename in os.listdir(image_dir):
        if not filename.endswith(('.jpg', '.jpeg', '.tiff')):
            continue
            
        image_path = os.path.join(image_dir, filename)
        print(f"\nProcessing: {filename}")
        
        # Get ground truth for this image
        gt_boxes = gt_labels[gt_labels['filename'] == filename]
        
        # Run model prediction
        results = model.predict(source=image_path, conf=0.25)
        
        # Load image for visualization
        img = cv2.imread(image_path)
        
        # Draw ground truth boxes in green
        for _, row in gt_boxes.iterrows():
            cv2.rectangle(img, 
                        (int(row['xmin']), int(row['ymin'])), 
                        (int(row['xmax']), int(row['ymax'])), 
                        (0, 255, 0), 2)  # Green for ground truth
            cv2.putText(img, f"GT: {class_metrics[row['class_id']]['name']}", 
                       (int(row['xmin']), int(row['ymin'])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Process predictions
        for result in results:
            boxes = result.boxes
            
            # Draw predicted boxes in blue
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                pred_box = box.tolist()
                pred_class = int(cls.item()) + 1  # Convert 0-based to 1-based class IDs
                
                cv2.rectangle(img, 
                            (int(pred_box[0]), int(pred_box[1])), 
                            (int(pred_box[2]), int(pred_box[3])), 
                            (255, 0, 0), 2)  # Blue for predictions
                cv2.putText(img, f"Pred: {class_metrics[pred_class]['name']} ({conf:.2f})", 
                           (int(pred_box[0]), int(pred_box[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Find matching ground truth box
                matched = False
                for _, gt_row in gt_boxes.iterrows():
                    gt_box = [gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax']]
                    if (gt_row['class_id'] == pred_class and 
                        calculate_iou(pred_box, gt_box) >= iou_threshold):
                        class_metrics[pred_class]['TP'] += 1
                        matched = True
                        break
                
                if not matched:
                    class_metrics[pred_class]['FP'] += 1
        
        # Count false negatives
        for _, gt_row in gt_boxes.iterrows():
            matched = False
            gt_box = [gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax']]
            for box, cls in zip(boxes.xyxy, boxes.cls):
                pred_box = box.tolist()
                pred_class = int(cls.item()) + 1
                if (gt_row['class_id'] == pred_class and 
                    calculate_iou(pred_box, gt_box) >= iou_threshold):
                    matched = True
                    break
            if not matched:
                class_metrics[gt_row['class_id']]['FN'] += 1
        
        # Show image with both ground truth and predictions
        cv2.imshow("Evaluation", img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"IoU Threshold: {iou_threshold}")
    print("\nPer-Class Metrics:")
    for class_id, metrics in class_metrics.items():
        tp = metrics['TP']
        fp = metrics['FP']
        fn = metrics['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{metrics['name']}:")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1-Score: {f1:.2f}")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")

if __name__ == "__main__":
    model_path = "runs/train/thermal_yolo/weights/best.pt"
    image_dir = "data/processed/val/images"  # Use validation set for evaluation
    label_file = "data/labels_test_16_bit.txt"  # Original label file
    
    evaluate_model(model_path, image_dir, label_file) 