import pandas as pd
import os
from PIL import Image

# Define your class mapping. For example, if:
# 1 -> Pedestrian, 2 -> Cyclist, 3 -> Car
CLASS_MAPPING = {
    1: 0,  # Pedestrian
    2: 1,  # Cyclist
    3: 2   # Car
}

def convert_labels(txt_file, images_dir, output_dir):
    """
    Reads bounding box data from the label file, converts them to YOLO format,
    and creates one .txt file per image in the output directory.
    """
    df = pd.read_csv(
        txt_file,
        sep=" ",
        names=["filename", "class_id", "xmin", "ymin", "xmax", "ymax"]
    )

    # Group rows by image file
    grouped = df.groupby("filename")
    os.makedirs(output_dir, exist_ok=True)

    for filename, group in grouped:
        image_path = os.path.join(images_dir, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        # Create a YOLO-format label file for this image
        label_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(output_dir, label_filename)

        with open(output_path, "w") as f:
            for _, row in group.iterrows():
                # Map the original class to YOLO class index
                yolo_class = CLASS_MAPPING[row["class_id"]]
                # Calculate normalized center coordinates and bbox dimensions
                x_center = ((row["xmin"] + row["xmax"]) / 2.0) / width
                y_center = ((row["ymin"] + row["ymax"]) / 2.0) / height
                bbox_w = (row["xmax"] - row["xmin"]) / width
                bbox_h = (row["ymax"] - row["ymin"]) / height

                # Write out in YOLO format: class x_center y_center width height
                f.write(f"{yolo_class} {x_center} {y_center} {bbox_w} {bbox_h}\n")
        print(f"[OK] Created {output_path}")

if __name__ == "__main__":
    # You can run this script on either label file.
    # For example, to convert the 16-bit labels:
    txt_file = "data/labels_test_16_bit.txt"
    images_dir = "data/images"
    output_dir = "data/yolo_labels"
    convert_labels(txt_file, images_dir, output_dir)

