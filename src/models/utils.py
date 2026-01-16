import torch.nn as nn

def _get_activation(activation_name):
    """ Returns the activation function based on the name provided. """
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }
    if activation_name.lower() not in activations:
        print(f"Unsupported activation function: {activation_name}")
        print("Defaulting to ReLU activation.")
    return activations.get(activation_name.lower(), nn.ReLU())



