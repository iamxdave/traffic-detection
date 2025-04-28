# scripts/data_rnn.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from glob import glob

class RNNDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.sequences = self._collect_sequences(root_dir)

    def _collect_sequences(self, root_dir):
        image_files = sorted(glob(os.path.join(root_dir, "**/*.jpg"), recursive=True))
        sequences = []
        for i in range(len(image_files) - self.sequence_length + 1):
            seq = image_files[i:i + self.sequence_length]
            sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_paths = self.sequences[idx]
        images = []
        class_ids = []

        for path in seq_paths:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            images.append(image)

            label_path = path.replace(".jpg", ".txt").replace(".png", ".txt")
            class_id = 0  # fallback
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        class_id = int(lines[0].split()[0])
            class_ids.append(class_id)

        # Shape: [sequence_length, C, H, W]
        images = torch.stack(images)
        # Use most common class in sequence (or do sequence-to-sequence)
        label = torch.mode(torch.tensor(class_ids))[0]
        return images, label
