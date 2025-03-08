from ultralytics import YOLO
import cv2
import numpy as np

def run_inference(model_path, image_path):
    print(f"Loading model from: {model_path}")
    print(f"Running inference on image: {image_path}")
    
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
        image_path="data/images/image_6.jpeg"
    )
