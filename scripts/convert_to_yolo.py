import pandas as pd
import os
from PIL import Image
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define your class mapping
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
    logger.info(f"Reading labels from: {txt_file}")
    logger.info(f"Looking for images in: {images_dir}")
    logger.info(f"Saving labels to: {output_dir}")

    # Read the label file
    try:
        df = pd.read_csv(
            txt_file,
            sep=" ",
            names=["filename", "class_id", "xmin", "ymin", "xmax", "ymax"]
        )
        logger.info(f"Successfully read {len(df)} annotations")
    except Exception as e:
        logger.error(f"Error reading label file: {e}")
        return

    # Group rows by image file
    grouped = df.groupby("filename")
    logger.info(f"Found annotations for {len(grouped)} unique images")

    # Create output directories
    train_labels = Path(output_dir) / "train/labels"
    val_labels = Path(output_dir) / "val/labels"
    train_labels.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    for filename, group in grouped:
        # Look for the image in the train and val directories
        image_path = None
        for subdir in ['train/images', 'val/images']:
            test_path = Path(images_dir) / subdir / filename
            if test_path.exists():
                image_path = test_path
                break
        
        if image_path is None:
            logger.warning(f"Could not find {filename} in any subdirectory")
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                logger.info(f"Processing {filename} ({width}x{height})")
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            continue

        # Determine output directory based on image location
        if "train" in str(image_path):
            label_dir = train_labels
        else:
            label_dir = val_labels

        # Create a YOLO-format label file for this image
        label_filename = Path(filename).stem + ".txt"
        output_path = label_dir / label_filename

        with open(output_path, "w") as f:
            for _, row in group.iterrows():
                # Map the original class to YOLO class index (subtract 1 since original classes start at 1)
                yolo_class = row["class_id"] - 1
                
                # Calculate normalized center coordinates and dimensions
                x_center = ((row["xmin"] + row["xmax"]) / 2.0) / width
                y_center = ((row["ymin"] + row["ymax"]) / 2.0) / height
                bbox_w = (row["xmax"] - row["xmin"]) / width
                bbox_h = (row["ymax"] - row["ymin"]) / height

                # Write out in YOLO format: class x_center y_center width height
                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")
                
        logger.info(f"Created {output_path}")

if __name__ == "__main__":
    # Convert the 16-bit labels
    txt_file = "data/labels_test_16_bit.txt"
    images_dir = "data/images"
    output_dir = "data/processed"
    convert_labels(txt_file, images_dir, output_dir)

