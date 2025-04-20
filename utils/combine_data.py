import os

DATASET_DIR = "dataset"

def get_image_paths(root_folder):
    """
    Collects all image file paths recursively from a given folder.
    """
    image_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith((".jpg", ".png")):  # Assuming images are in JPG or PNG format
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def create_train_test_files():
    """
    Collects train and test images from the dataset and saves their paths to train.txt and test.txt.
    """
    train_images = []
    test_images = []

    # Collect train images from day and night
    train_images.extend(get_image_paths(os.path.join(DATASET_DIR, "day/train")))
    train_images.extend(get_image_paths(os.path.join(DATASET_DIR, "night/train")))

    # Collect test images from day and night
    test_images.extend(get_image_paths(os.path.join(DATASET_DIR, "day/test")))
    test_images.extend(get_image_paths(os.path.join(DATASET_DIR, "night/test")))

    # Save to files
    with open("dataset/train.txt", "w") as f:
        f.write("\n".join(train_images))

    with open("dataset/test.txt", "w") as f:
        f.write("\n".join(test_images))

    print(f"✅ Train and test files created! {len(train_images)} train images, {len(test_images)} test images.")

if __name__ == "__main__":
    create_train_test_files()
