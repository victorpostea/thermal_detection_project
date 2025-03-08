from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import requests
from io import BytesIO
import os

app = Flask(__name__)

def preprocess_thermal_image(image):
    """
    Preprocess thermal image data for YOLO inference.
    """
    # Print original image info for debugging
    print(f"Original image info - Shape: {image.shape}, dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
    
    # If it's a 16-bit image, normalize to 8-bit
    if image.dtype == np.uint16:
        # Normalize to 0-255 range
        img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = img_normalized.astype(np.uint8)
        
        # Apply color map for better visualization (COLORMAP_INFERNO is good for thermal)
        image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
    
    return image

# Define model path
MODEL_PATH = "runs/train/thermal_yolo/weights/best.pt"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find thermal model at {MODEL_PATH}. Please ensure the model file exists.")

# Initialize the YOLO model globally with our custom thermal model
print(f"Loading thermal detection model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Thermal detection system is running",
        "model_path": MODEL_PATH
    })

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Check if URL is provided in the request
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "No image URL provided"}), 400

        # Get image from URL
        response = requests.get(data['image_url'])
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL"}), 400

        # Convert image to format suitable for OpenCV
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)  # Changed to UNCHANGED to preserve bit depth
        
        # Preprocess the image if it's thermal
        image = preprocess_thermal_image(image)

        # Run inference
        results = model.predict(image, conf=0.30)  # Updated confidence threshold to match inference.py
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                detections.append({
                    "bbox": box.tolist(),
                    "class": int(cls.item()),
                    "confidence": float(conf.item())
                })

        return jsonify({
            "status": "success",
            "detections": detections,
            "class_mapping": {
                0: "person",
                1: "bicycle/bike",
                2: "vehicle/car"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 