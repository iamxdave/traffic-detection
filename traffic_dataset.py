import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from collections import defaultdict


class TrafficDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms if transforms else T.ToTensor()

        self.grouped_data = defaultdict(list)
        for _, row in self.annotations.iterrows():
            path = row['Filename']
            self.grouped_data[path].append([
                row['X_min'], row['Y_min'], row['X_max'], row['Y_max'], row['Class ID']
            ])
        self.image_paths = list(self.grouped_data.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        objects = self.grouped_data[img_path]
        boxes = torch.tensor([o[:4] for o in objects], dtype=torch.float32)
        labels = torch.tensor([o[4] for o in objects], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        img = self.transforms(img) if callable(self.transforms) else T.ToTensor()(img)
        return img, target
