from ultralytics import YOLO
import cv2

def run_inference(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.5)

    # Print and visualize results
    for result in results:
        print("Bounding boxes:", result.boxes.xyxy)
        print("Classes:", result.boxes.cls)
        print("Confidences:", result.boxes.conf)
        annotated_frame = result.plot()  # Annotated image with boxes
        cv2.imshow("Detections", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(
        model_path="runs/detect/thermal_yolo7/weights/best.pt",
        image_path="data/images/image_8.jpeg"  # Test on a TIFF image or JPEG
    )
