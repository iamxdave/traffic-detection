import os
import pandas as pd

DATASET_DIR = "dataset"

# Mapping annotation names to YOLO class names
CLASS_MAPPING = {
    "go": "MG", "stop": "MR",
    "stopLeft": "LR", "goLeft": "LG",
    "warning": "MY", "warningLeft": "LY"
}

def convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max):
    """
    Converts bounding box coordinates to YOLO format:
    class_id x_center y_center width height (normalized between 0 and 1)
    """
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

def process_annotations(annotation_file, image_folder):
    """
    Reads CSV annotation file and converts labels to YOLO format.
    """
    df = pd.read_csv(annotation_file, delimiter=";")
    
    for _, row in df.iterrows():
        filename = row["Filename"].split("/")[-1]  # Extract image filename
        class_name = row["Annotation tag"]
        
        if class_name not in CLASS_MAPPING:
            continue  # Ignore unknown labels

        # Get bounding box coordinates
        x_min, y_min, x_max, y_max = row["Upper left corner X"], row["Upper left corner Y"], row["Lower right corner X"], row["Lower right corner Y"]

        # Load image size dynamically (assuming all images exist in frames/)
        img_path = os.path.join(image_folder, "frames", filename)
        if not os.path.exists(img_path):
            continue

        img_width, img_height = 1280, 960  # LISA dataset standard resolution

        # Convert to YOLO format
        yolo_box = convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max)
        class_id = list(CLASS_MAPPING.keys()).index(class_name)  # Assign class ID

        # Save label file (same name as image, but with .txt)
        label_path = os.path.join(image_folder, "frames", filename.replace(".jpg", ".txt"))
        with open(label_path, "a") as f:
            f.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

def process_dataset():
    """
    Walk through dataset folders and process each annotation file.
    """
    for mode in ["day", "night"]:
        for dataset_type in ["train", "test"]:
            dataset_path = os.path.join(DATASET_DIR, mode, dataset_type)

            for clip_folder in os.listdir(dataset_path):
                clip_path = os.path.join(dataset_path, clip_folder)
                annotation_box_file = os.path.join(clip_path, "frameAnnotationsBOX.csv")
                annotation_bulb_file = os.path.join(clip_path, "frameAnnotationsBULB.csv")

                if os.path.exists(annotation_box_file):
                    process_annotations(annotation_box_file, clip_path)
                
                if os.path.exists(annotation_bulb_file):
                    process_annotations(annotation_bulb_file, clip_path)

    print("✅ Labels converted successfully!")

if __name__ == "__main__":
    process_dataset()
