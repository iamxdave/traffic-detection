# scripts/data_cnn.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CNNDataset(Dataset):
    def __init__(self, list_file, transform=None):
        with open(list_file, 'r') as f:
            self.image_paths = f.read().splitlines()
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        class_id = 0  # fallback
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    class_id = int(lines[0].split()[0])  # take first object class

        return image, torch.tensor(class_id)
