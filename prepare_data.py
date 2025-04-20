import os
import csv
import pandas as pd

DATASET_DIR = "dataset"
OUTPUT_CSV = os.path.join(DATASET_DIR, "annotations.csv")

CLASS_MAPPING = {
    "go": 0, "goLeft": 1,
    "stop": 2, "stopLeft": 3,
    "warning": 4, "warningLeft": 5
}

def get_image_paths(root_folder):
    """ Collects all image file paths recursively from a given folder. """
    image_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith((".jpg", ".png")):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max):
    """ Converts bounding box coordinates to YOLO format (normalized). """
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_annotations(annotation_file, image_folder, csv_writer):
    """ Reads annotation CSV and converts labels to YOLO format. """
    df = pd.read_csv(annotation_file, delimiter=";")

    for _, row in df.iterrows():
        filename = os.path.basename(row["Filename"])
        class_name = row["Annotation tag"]

        if class_name not in CLASS_MAPPING:
            continue  

        x_min, y_min, x_max, y_max = row["Upper left corner X"], row["Upper left corner Y"], row["Lower right corner X"], row["Lower right corner Y"]
        img_path = os.path.join(image_folder, "frames", filename)
        
        if not os.path.exists(img_path):
            continue  

        img_width, img_height = 1280, 960  
        yolo_bbox = convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max)
        class_id = CLASS_MAPPING[class_name]

        # Save YOLO format labels
        label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")
        with open(label_path, "a") as f:
            f.write(f"{class_id} {yolo_bbox}\n")

        # Save CSV format (for future models)
        csv_writer.writerow([img_path, class_id, x_min, y_min, x_max, y_max])

def create_train_test_files():
    """ Generates train.txt and test.txt with image paths & creates annotations.csv """
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

                    process_annotations(annotation_file, clip_path, csv_writer)

                    # Collect image paths
                    images = get_image_paths(os.path.join(clip_path, "frames"))
                    if dataset_type == "train":
                        train_images.extend(images)
                    else:
                        test_images.extend(images)

    # Save image lists for YOLO training
    with open(os.path.join(DATASET_DIR, "train.txt"), "w") as f:
        f.write("\n".join(train_images))
    with open(os.path.join(DATASET_DIR, "test.txt"), "w") as f:
        f.write("\n".join(test_images))

    print(f"✅ Data prepared! {len(train_images)} train images, {len(test_images)} test images.")

if __name__ == "__main__":
    create_train_test_files()
