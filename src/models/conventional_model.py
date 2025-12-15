import torch
import torch.nn as nn

class ConventionalNN(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config['conventional_nn_model']
        self.input_size = input_size
        self.model_name = type(self).__name__ 

        # Build MLP layers dynamically from the config
        mlp_layers = []
        in_features = self.input_size
        for layer_params in self.config['mlp_layers']:
            mlp_layers.append(nn.Linear(in_features, layer_params['units']))

            # Add BatchNorm only if enabled in config
            if layer_params.get('batchnorm', False):
                mlp_layers.append(nn.BatchNorm1d(layer_params['units']))
            # Activation function
            activation_name = layer_params['activation']
            if activation_name == 'ReLU':
                mlp_layers.append(nn.ReLU())
            elif activation_name == 'GELU':
                mlp_layers.append(nn.GELU())
            elif activation_name == 'Sigmoid':
                mlp_layers.append(nn.Sigmoid())
            elif activation_name == 'Tanh':
                mlp_layers.append(nn.Tanh())

            # Dropoout if specified
            if 'dropout_rate' in layer_params:
                mlp_layers.append(nn.Dropout(layer_params['dropout_rate']))

            in_features = layer_params['units']

        # Final output layer
        mlp_layers.append(nn.Linear(in_features, self.config['output_size']))

        self.mlp_stack = nn.Sequential(*mlp_layers)

    def forward(self, x):

        logits = self.mlp_stack(x)
        return logits



