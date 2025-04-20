# train_model.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn_model import CNN
from models.yolo_model import YOLO
from models.rnn_model import RNN
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def build_model(config):
    if config['model_type'] == 'yolo':
        model = YOLO(config)
    elif config['model_type'] == 'cnn':
        model = CNN(config)
    elif config['model_type'] == 'rnn':
        model = RNN(config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    return model

def train_model(model, config, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    model.train()
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
