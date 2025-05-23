import os
import csv
import pandas as pd
from pathlib import Path
from random import choices

# Define the base dataset directory
DATASET_DIR = Path("dataset")
OUTPUT_CSV = DATASET_DIR / "annotations.csv"

# Mapping of class names to class IDs
CLASS_MAPPING = {
    "go": 0, "goLeft": 1,
    "stop": 2, "stopLeft": 3,
    "warning": 4, "warningLeft": 5
}

def convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_annotations(annotation_file, image_folder, csv_writer):
    df = pd.read_csv(annotation_file, delimiter=";")
    valid_images = set()

    frames_folder = Path(image_folder) / "frames"
    for fname in frames_folder.glob("*.txt"):
        fname.unlink()

    for _, row in df.iterrows():
        filename = Path(row["Filename"]).name
        class_name = row["Annotation tag"]

        if class_name not in CLASS_MAPPING:
            continue

        x_min = row["Upper left corner X"]
        y_min = row["Upper left corner Y"]
        x_max = row["Lower right corner X"]
        y_max = row["Lower right corner Y"]

        img_path = frames_folder / filename
        if not img_path.exists():
            continue

        img_width, img_height = 1280, 960
        yolo_bbox = convert_to_yolo_format(img_width, img_height, x_min, y_min, x_max, y_max)
        class_id = CLASS_MAPPING[class_name]

        label_path = img_path.with_suffix(".txt")
        with open(label_path, "a") as f:
            f.write(f"{class_id} {yolo_bbox}\n")

        csv_writer.writerow([img_path.as_posix(), class_id, x_min, y_min, x_max, y_max])
        valid_images.add(img_path.as_posix())

    return valid_images

def create_train_test_files():
    train_images, test_images = [], []

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Class ID", "X_min", "Y_min", "X_max", "Y_max"])

        for mode in ["day", "night"]:
            for dataset_type in ["train", "test"]:
                dataset_path = DATASET_DIR / mode / dataset_type

                for clip_folder in os.listdir(dataset_path):
                    clip_path = dataset_path / clip_folder
                    annotation_file = clip_path / "frameAnnotationsBOX.csv"

                    if not annotation_file.exists():
                        continue

                    valid_images = process_annotations(annotation_file, clip_path, csv_writer)

                    if dataset_type == "train":
                        train_images.extend(valid_images)
                    else:
                        test_images.extend(valid_images)

    def get_valid_images(images):
        return [img for img in images if Path(img).with_suffix(".txt").exists()]

    def write_list_to_file(filepath, paths):
        with open(filepath, "w") as f:
            f.write("\n".join(sorted(paths)))

    train_images = get_valid_images(train_images)
    test_images = get_valid_images(test_images)

    write_list_to_file(DATASET_DIR / "train.txt", train_images)
    write_list_to_file(DATASET_DIR / "test.txt", test_images)

    subsets = {
        "train_day.txt": [p for p in train_images if "/day/" in p],
        "train_night.txt": [p for p in train_images if "/night/" in p],
        "test_day.txt": [p for p in test_images if "/day/" in p],
        "test_night.txt": [p for p in test_images if "/night/" in p],
    }

    for filename, paths in subsets.items():
        write_list_to_file(DATASET_DIR / filename, paths)

    df = pd.read_csv(OUTPUT_CSV)
    df_train = df[df['Filename'].str.contains('/train/')]
    df_test = df[df['Filename'].str.contains('/test/')]
    df_train.to_csv(DATASET_DIR / "annotations_train.csv", index=False)
    df_test.to_csv(DATASET_DIR / "annotations_test.csv", index=False)

    print(f"✅ Data prepared! {len(train_images)} valid train images, {len(test_images)} valid test images.")
    print(f"📄 Saved: train/test .txt and annotations .csv files")

def remove_images_without_labels():
    for mode in ["day", "night"]:
        for dataset_type in ["train", "test"]:
            dataset_path = DATASET_DIR / mode / dataset_type

            for clip_folder in os.listdir(dataset_path):
                frames_path = dataset_path / clip_folder / "frames"
                if frames_path.exists():
                    for img_filename in os.listdir(frames_path):
                        img_path = frames_path / img_filename
                        label_path = img_path.with_suffix(".txt")
                        if not label_path.exists():
                            print(f"Removing image: {img_path}")
                            img_path.unlink()

def balance_data():
    annotations_path = DATASET_DIR / "annotations.csv"
    balanced_csv_path = DATASET_DIR / "annotations_balanced.csv"

    df = pd.read_csv(annotations_path)
    class_counts = df['Class ID'].value_counts()
    max_count = class_counts.max()
    balanced_rows = []
    for class_id in class_counts.index:
        class_df = df[df['Class ID'] == class_id]
        repeat = max_count // len(class_df)
        remainder = max_count % len(class_df)
        balanced_rows.extend([*class_df.to_dict("records")] * repeat)
        balanced_rows.extend(choices(class_df.to_dict("records"), k=remainder))

    df_balanced = pd.DataFrame(balanced_rows)
    df_balanced.to_csv(balanced_csv_path, index=False)

    df_balanced_train = df_balanced[df_balanced['Filename'].str.contains('/train/')]
    df_balanced_test = df_balanced[df_balanced['Filename'].str.contains('/test/')]
    df_balanced_train.to_csv(DATASET_DIR / "annotations_train_balanced.csv", index=False)
    df_balanced_test.to_csv(DATASET_DIR / "annotations_test_balanced.csv", index=False)

    subsets = {
        "train_balanced.txt": df_balanced_train,
        "test_balanced.txt": df_balanced_test,
        "train_balanced_day.txt": df_balanced_train[df_balanced_train['Filename'].str.contains('/day/')],
        "train_balanced_night.txt": df_balanced_train[df_balanced_train['Filename'].str.contains('/night/')],
        "test_balanced_day.txt": df_balanced_test[df_balanced_test['Filename'].str.contains('/day/')],
        "test_balanced_night.txt": df_balanced_test[df_balanced_test['Filename'].str.contains('/night/')],
    }

    for name, df_subset in subsets.items():
        paths = df_subset['Filename'].unique()
        with open(DATASET_DIR / name, 'w') as f:
            f.write("\n".join(sorted(paths)))

    print("✅ Balanced splits and .txt lists created.")

if __name__ == "__main__":
    create_train_test_files()
    remove_images_without_labels()
    balance_data()
