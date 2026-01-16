import torch
import torch.nn as nn

from models.utils import _get_activation

class ConventionalNN(nn.Module):
    """
    Configurable Multi-Layer Perceptron (MLP) for binary/multi-class classification.
    
    Dynamically constructs a feedforward neural network from configuration parameters,
    supporting variable layer sizes, activation functions, batch normalization, and dropout.
    """
    def __init__(self, input_size, config):
        super().__init__()
        self.config = config['conventional_nn_model']
        self.input_size = input_size
        self.model_name = type(self).__name__ 

        # Build MLP layers dynamically from the config
        mlp_layers = []
        in_features = self.input_size
        for layer_params in self.config['mlp_layers']:
            units =  layer_params.get('units', 64)
            activation = _get_activation(layer_params.get('activation', 'ReLU'))
            dropout_rate = layer_params.get('dropout_rate', 0.0)
            batchnorm = layer_params.get('batchnorm', False)

            mlp_layers.append(nn.Linear(in_features, units))

            # Add BatchNorm only if enabled in config
            if batchnorm:
                mlp_layers.append(nn.BatchNorm1d(units))
            # Activation function
            mlp_layers.append(activation)

            # Dropoout if specified
            if dropout_rate > 0.0:
                mlp_layers.append(nn.Dropout(dropout_rate))

            in_features = units

        # Final output layer
        mlp_layers.append(nn.Linear(in_features, self.config['output_size']))

        self.mlp_stack = nn.Sequential(*mlp_layers)

    def forward(self, x):
        """ Returns the logits """
        if x.dim() == 3:
            x = x.mean(dim=1)  
        logits = self.mlp_stack(x)
        return logits
    
    def predict_proba(self, x):
        """ Returns the probabilities after applying sigmoid """
        return torch.sigmoid(self.forward(x))



