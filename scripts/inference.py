from ultralytics import YOLO
import cv2
import numpy as np

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

def run_inference(model_path, image_path):
    print(f"Loading model from: {model_path}")
    print(f"Running inference on image: {image_path}")
    
    # Preprocess the image if it's a TIFF file
    if image_path.lower().endswith('.tiff') or image_path.lower().endswith('.tif'):
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
    results = model.predict(source=image_path, conf=0.25)  # Lowered confidence threshold

    # Print and visualize results
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("No detections found!")
        else:
            print(f"Found {len(boxes)} detections:")
            print("Bounding boxes:", boxes.xyxy.tolist())
            print("Classes:", boxes.cls.tolist())
            print("Confidences:", boxes.conf.tolist())
        
        # Get the annotated frame
        annotated_frame = result.plot()
        
        # Display image dimensions and type for debugging
        print(f"Image shape: {annotated_frame.shape}")
        print(f"Image dtype: {annotated_frame.dtype}")
        
        cv2.imshow("Detections", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(
        model_path="yolov8n.pt",  # Using the model in root directory
        image_path="data/images/image_1.tiff"  # Try with a TIFF file
    )
