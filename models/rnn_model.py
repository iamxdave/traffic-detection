import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config = config
        # Define the RNN layers dynamically based on the config file
        self.rnn_layers = self.build_rnn_layers(config)

    def build_rnn_layers(self, config):
        layers = []
        
        for layer in config['layers']:
            if layer['type'] == 'rnn':
                layers.append(nn.RNN(input_size=layer['input_size'],
                                     hidden_size=layer['hidden_size'],
                                     num_layers=layer['num_layers'],
                                     batch_first=layer['batch_first']))
                if 'activation' in layer:
                    layers.append(nn.ReLU() if layer['activation'] == 'relu' else nn.LeakyReLU())
            elif layer['type'] == 'lstm':
                layers.append(nn.LSTM(input_size=layer['input_size'],
                                      hidden_size=layer['hidden_size'],
                                      num_layers=layer['num_layers'],
                                      batch_first=layer['batch_first']))
                if 'activation' in layer:
                    layers.append(nn.ReLU() if layer['activation'] == 'relu' else nn.LeakyReLU())
            elif layer['type'] == 'fully_connected':
                layers.append(nn.Linear(in_features=layer['in_features'], out_features=layer['out_features']))
                if 'activation' in layer:
                    layers.append(nn.ReLU() if layer['activation'] == 'relu' else nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.rnn_layers(x)
