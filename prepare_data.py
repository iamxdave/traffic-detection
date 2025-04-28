import os
import csv
import pandas as pd

# Define the base dataset directory
DATASET_DIR = "dataset"
OUTPUT_CSV = os.path.join(DATASET_DIR, "annotations.csv")

# Mapping of class names to class IDs
CLASS_MAPPING = {
    "go": 0, "goLeft": 1,
    "stop": 2, "stopLeft": 3,
    "warning": 4, "warningLeft": 5
}

def convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max):
    """Convert bounding box coordinates to YOLO format."""
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_annotations(annotation_file, image_folder, csv_writer):
    """Process annotations and write YOLO-formatted labels and CSV entries."""
    df = pd.read_csv(annotation_file, delimiter=";")
    valid_images = set()

    for _, row in df.iterrows():
        filename = os.path.basename(row["Filename"])
        class_name = row["Annotation tag"]

        if class_name not in CLASS_MAPPING:
            continue  # Skip unrecognized classes

        x_min = row["Upper left corner X"]
        y_min = row["Upper left corner Y"]
        x_max = row["Lower right corner X"]
        y_max = row["Lower right corner Y"]

        img_path = os.path.join(image_folder, "frames", filename)

        if not os.path.exists(img_path):
            continue

        img_width, img_height = 1280, 960  # Replace with actual dimensions if different
        yolo_bbox = convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max)
        class_id = CLASS_MAPPING[class_name]

        label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")
        with open(label_path, "a") as f:
            f.write(f"{class_id} {yolo_bbox}\n")

        csv_writer.writerow([img_path, class_id, x_min, y_min, x_max, y_max])
        valid_images.add(img_path)  # Only add if annotation was valid

    return valid_images

def create_train_test_files():
    """Create train.txt and test.txt files with correct image paths."""
    train_images, test_images = [], []

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Class ID", "X_min", "Y_min", "X_max", "Y_max"])

        for mode in ["day", "night"]:
            for dataset_type in ["train", "test"]:
                dataset_path = os.path.join(DATASET_DIR, mode, dataset_type)

                for clip_folder in os.listdir(dataset_path):
                    clip_path = os.path.join(dataset_path, clip_folder)
                    annotation_file = os.path.join(clip_path, "frameAnnotationsBOX.csv")

                    if not os.path.exists(annotation_file):
                        continue

                    valid_images = process_annotations(annotation_file, clip_path, csv_writer)

                    if dataset_type == "train":
                        train_images.extend(valid_images)
                    else:
                        test_images.extend(valid_images)

    # Write the image paths to train.txt and test.txt
    with open(os.path.join(DATASET_DIR, "train.txt"), "w") as f:
        f.write("\n".join(sorted(train_images)))
    with open(os.path.join(DATASET_DIR, "test.txt"), "w") as f:
        f.write("\n".join(sorted(test_images)))

    print(f"✅ Data prepared! {len(train_images)} valid train images, {len(test_images)} valid test images.")

if __name__ == "__main__":
    create_train_test_files()
