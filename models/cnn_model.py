import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        # Define the model layers dynamically based on the config file
        self.layers = self.build_layers(config)

    def build_layers(self, config):
        layers = []
        # Build layers based on the config file (this is just an example)
        for layer in config['layers']:
            if layer['type'] == 'convolutional':
                layers.append(nn.Conv2d(in_channels=layer['in_channels'],
                                        out_channels=layer['out_channels'],
                                        kernel_size=layer['kernel_size'],
                                        stride=layer['stride'],
                                        padding=layer['padding']))
                if 'batch_norm' in layer and layer['batch_norm']:
                    layers.append(nn.BatchNorm2d(layer['out_channels']))
                if 'activation' in layer:
                    layers.append(nn.ReLU() if layer['activation'] == 'relu' else nn.LeakyReLU())
            elif layer['type'] == 'fully_connected':
                layers.append(nn.Linear(in_features=layer['in_features'], out_features=layer['out_features']))
                if 'activation' in layer:
                    layers.append(nn.ReLU() if layer['activation'] == 'relu' else nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
