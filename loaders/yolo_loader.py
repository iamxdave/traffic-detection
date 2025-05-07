import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class YOLODataset(Dataset):
    def __init__(self, txt_file, transform=None, img_size=(608, 608)):
        self.imgs = []
        self.labels = []
        self.img_size = img_size  # (width, height)

        # Read the txt file with image paths
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            img_path = line.strip()
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(txt_file), img_path)
            img_path = os.path.normpath(img_path)
            self.imgs.append(img_path)
            self.labels.append(img_path.replace('.jpg', '.txt'))

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
        print(f"Fetching item at index {idx}")

        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        print(f"Image shape: {image.shape}")

        # Load and process label
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                boxes.append(parts)  # assuming format [class, x_center, y_center, width, height]

        if len(boxes) == 0:
            boxes = torch.zeros((0, 5))  # No objects in image
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        print(f"Boxes shape: {boxes.shape}")

        return image, boxes
    
    def collate_fn(self, batch):
        images, targets = zip(*batch)
    
        images = torch.stack(images, 0)

        # Keep targets as list of tensors (bounding boxes can vary in number)
        targets = list(targets)

        return images, targets

