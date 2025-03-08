from ultralytics import YOLO

def train_yolo(data_yaml, epochs=50, model_type="yolov8n.pt"):
    model = YOLO(model_type)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,
        name="thermal_yolo",
    )

if __name__ == "__main__":
    train_yolo("data/data.yaml", epochs=50, model_type="yolov8n.pt")
