import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class YOLODataset(Dataset):
    def __init__(self, txt_file, transform=None, img_size=(608, 608)):
        self.imgs = []
        self.labels = []
        self.img_size = img_size  # (width, height)

        txt_path = Path(txt_file).resolve()

        # Read the txt file with image paths
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            img_rel_path = line.strip()
            
            # Check if we're on a Windows machine and adjust paths
            if os.name == 'nt':  # Windows
                img_rel_path = img_rel_path.replace('/', '\\')
            else:  # Unix-based OS (Mac/Linux)
                img_rel_path = img_rel_path.replace('\\', '/')

            # Construct the absolute path
            img_path = (txt_path.parent / img_rel_path).resolve()
            label_path = img_path.with_suffix('.txt')
            self.imgs.append(img_path)
            self.labels.append(label_path)

        print(f"Loaded {len(self.imgs)} images and {len(self.labels)} labels")

        if len(self.imgs) == 0:
            raise ValueError(f"Dataset is empty, check your paths or txt file: {txt_file}")

        # Default transform if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),  # Convert to [0,1] tensor
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label_path = self.labels[idx]

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Load and process label
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    boxes.append(parts)

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return image, boxes

    def collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        targets = list(targets)  # bounding boxes are variable
        return images, targets
