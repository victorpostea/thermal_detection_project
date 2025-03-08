from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse

def preprocess_thermal_image(image_path):
    """
    Preprocess 16-bit TIFF thermal images for YOLO inference.
    """
    # Read the image in its original format
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Print original image info for debugging
    print(f"Original image info - Shape: {img.shape}, dtype: {img.dtype}, min: {np.min(img)}, max: {np.max(img)}")
    
    # If it's a 16-bit image, normalize to 8-bit
    if img.dtype == np.uint16:
        # Normalize to 0-255 range
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img_normalized.astype(np.uint8)
        
        # Apply color map for better visualization (COLORMAP_INFERNO is good for thermal)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    
    return img

def run_inference(model_path, image_path, conf_threshold=0.25):
    # Check if the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        # Try changing extension from .tiff to .jpg if it doesn't exist
        if image_path.lower().endswith(('.tiff', '.tif')):
            jpg_path = image_path[:-5] + '.jpg'  # Remove .tiff or .tif and add .jpg
            if os.path.exists(jpg_path):
                print(f"Original image not found at {image_path}, using processed version at {jpg_path}")
                image_path = jpg_path
            else:
                raise FileNotFoundError(f"Image not found at: {image_path} or {jpg_path}")
        else:
            raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"Loading model from: {model_path}")
    print(f"Running inference on image: {image_path}")
    
    # Only preprocess if it's a TIFF file
    if image_path.lower().endswith(('.tiff', '.tif')):
        try:
            preprocessed_img = preprocess_thermal_image(image_path)
            # Save preprocessed image temporarily
            temp_path = 'temp_processed.jpg'
            cv2.imwrite(temp_path, preprocessed_img)
            image_path = temp_path
        except Exception as e:
            print(f"Error preprocessing thermal image: {e}")
            return
    
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf_threshold)

    # Print and visualize results
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("No detections found!")
        else:
            print(f"\nFound {len(boxes)} detections:")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"Detection {i+1}:")
                print(f"  Class: {model.names[cls]}")
                print(f"  Confidence: {conf:.2f}")
                print(f"  Bounding box: {box.xyxy[0].tolist()}")
        
        # Get the annotated frame
        annotated_frame = result.plot()
        
        # Display image dimensions and type for debugging
        print(f"\nImage shape: {annotated_frame.shape}")
        print(f"Image dtype: {annotated_frame.dtype}")
        
        # Show the image
        cv2.imshow("Detections", annotated_frame)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on thermal images')
    parser.add_argument('--source', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default="runs/train/thermal_yolo/weights/best.pt", 
                      help='Path to the model weights')
    parser.add_argument('--conf', type=float, default=0.25, 
                      help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        image_path=args.source,
        conf_threshold=args.conf
    )
