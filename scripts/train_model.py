# scripts/train_model.py

import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from functools import partial

# import models
from models.yolo_model import YOLO
from models.cnn_model  import CNN
from models.rnn_model  import RNN

# import loaders
from loaders.yolo_loader import YOLODataset
from loaders.cnn_loader  import CNNDataset
from loaders.rnn_loader  import RNNDataset
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def build_model(config):
    t = config['model_type']
    if   t == 'yolo': return YOLO(config)
    elif t == 'cnn':  return CNN(config)
    elif t == 'rnn':  return RNN(config)
    else: raise ValueError(f"Unknown model_type {t}")

def build_dataloader(config, dataset_config, train=True):
    t = config['model_type']
    dataset_base_path = dataset_config['path']
    list_file = dataset_config['train'] if train else dataset_config['val']
    list_file_path = os.path.join(PROJECT_ROOT, dataset_base_path, list_file)
    
    transform = T.Compose([
        T.Resize((config['height'], config['width'])),
        T.ToTensor()
    ])

    if t == 'yolo':
        dataset = YOLODataset(list_file_path, transform=transform)
    elif t == 'cnn':
        dataset = CNNDataset(list_file_path, transform=transform)
    elif t == 'rnn':
        root = dataset_base_path
        dataset = RNNDataset(root, sequence_length=config.get('sequence_length', 5), transform=transform)
    else:
        raise ValueError(f"Unknown model_type {t}")
    
    collate_fn = partial(dataset.collate_fn, subdivisions=config.get('subdivisions', 1))

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=train,
        num_workers=3,
        pin_memory=True,
        collate_fn=collate_fn if callable(collate_fn) else None
    )

    print(f"DataLoader created with batch size: {config['batch_size']}")  # Debugowanie DataLoadera
    return dataloader

# Training function with early stopping
def train_model_with_early_stopping(model, config, train_loader, val_loader, patience=10, min_delta=0.001, save_path='best_model.pth'):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    subdivisions = config.get('subdivisions', 1)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images, labels = batch

            # Jeśli subdivisions > 1, images to tuple -> sklej
            if isinstance(images, (tuple, list)):
                images = torch.cat(images, dim=0)
            images = images.to(device)

            # Labels - lista tensorów
            if isinstance(labels, (tuple, list)):
                labels = [label[0] if isinstance(label, tuple) else label for label in labels]  # odpakuj jeśli tuple
                labels = torch.cat(labels, dim=0) if isinstance(labels[0], torch.Tensor) else labels
                labels = labels.to(device) if isinstance(labels, torch.Tensor) else labels
            else:
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_images, val_labels = val_batch

                if isinstance(val_images, (tuple, list)):
                    val_images = torch.cat(val_images, dim=0)
                val_images = val_images.to(device)

                if isinstance(val_labels, (tuple, list)):
                    val_labels = [label[0] if isinstance(label, tuple) else label for label in val_labels]
                    val_labels = torch.cat(val_labels, dim=0) if isinstance(val_labels[0], torch.Tensor) else val_labels
                    val_labels = val_labels.to(device) if isinstance(val_labels, torch.Tensor) else val_labels
                else:
                    val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += compute_loss(val_outputs, val_labels, criterion).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model
    model.load_state_dict(torch.load(save_path))
    return model


def compute_loss(outputs, labels, criterion):
    """
    Oblicza stratę dla modelu. Funkcja musi być dostosowana do specyficznego przypadku,
    np. detekcji obiektów, gdzie `labels` może być listą tensorów.
    """
    # Przykład dla klasyfikacji (dostosuj do detekcji obiektów)
    if isinstance(labels, list):
        # Jeśli labels to lista, przekształć ją w odpowiedni format
        labels = torch.cat(labels, dim=0)  # Połącz listę w jeden tensor, jeśli to możliwe
    return criterion(outputs, labels)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
