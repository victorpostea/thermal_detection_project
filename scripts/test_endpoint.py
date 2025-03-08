import requests
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os
import time
import json

# Start a simple HTTP server to serve our test image
def start_image_server():
    os.chdir("data/images")  # Change to the images directory
    httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    return httpd

def test_detection():
    # Start the image server
    image_server = start_image_server()
    
    try:
        # Give the servers a moment to start
        time.sleep(1)
        
        # Test image URL (using one of your local thermal images)
        image_url = "http://localhost:8000/image_8.jpeg"
        
        # Make request to detection endpoint
        response = requests.post(
            "http://localhost:5000/detect",
            json={"image_url": image_url}
        )
        
        # Print results
        print("\nDetection Results:")
        print("Status Code:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("\nDetections:")
            for detection in result["detections"]:
                class_id = detection["class"]
                class_name = result["class_mapping"][str(class_id)]
                confidence = detection["confidence"]
                bbox = detection["bbox"]
                print(f"\nClass: {class_name}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Bounding Box: {bbox}")
        else:
            print("Error:", response.text)
            
    finally:
        # Cleanup
        image_server.shutdown()
        image_server.server_close()

if __name__ == "__main__":
    test_detection() 