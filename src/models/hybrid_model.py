import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron

class HybridModel(nn.Module):
    def __init__(self, snn_input_size, snn_time_steps, config):
        """
        A hybrid model combining a Spiking Neural Network (SNN) and a Conventional Neural Network (CNN).
        The SNN and CNN operate in parallel on the same input, and their outputs are fused for
        final classification.
        
        Args:
            snn_input_size (int): The number of features in the input data for the SNN.
            snn_time_steps (int): The number of time steps for the SNN simulation.
            config (dict): A dictionary containing model configurations.
        """
        super().__init__()
        self.snn_time_steps = snn_time_steps
        snn_config = config['snn_model']
        conv_config = config['conventional_nn_model']
        self.model_name = type(self).__name__ 
        
        # Get the SNN hidden size from the 'hidden_layers' list in the config
        snn_hidden_size = snn_config['hidden_layers'][-1]['units']
        conv_hidden_size = conv_config['mlp_layers'][-1]['units']
        
        # SNN part: acts as one of the feature extractors
        # This is the SNN network which processes the sequential data
        self.snn_model = nn.Sequential(
            nn.Linear(snn_input_size, snn_config['hidden_layers'][-1]['units']),
            neuron.LIFNode(tau=2.0)
        )
        
        # Conventional NN part: acts as the other feature extractor
        # This is the conventional network which processes the last time-step of the data
        self.conv_model = nn.Sequential(
            nn.Linear(snn_input_size, conv_config['mlp_layers'][0]['units']),
            nn.ReLU(),
            nn.Linear(conv_config['mlp_layers'][0]['units'], conv_hidden_size),
            nn.ReLU()
        )
        
        # Fusion layer and final output
        # The input size is the sum of the hidden sizes from both models
        self.fusion_layer = nn.Linear(snn_hidden_size + conv_hidden_size, 1)

    def forward(self, x):
        """
        Performs the forward pass for the hybrid model.
        
        Args:
            x (torch.Tensor): The input data tensor.
        
        Returns:
            torch.Tensor: The output of the fusion layer.
        """
        # Reset the SNN's state before the forward pass
        functional.reset_net(self.snn_model)
        
        # Split input for SNN and Conventional NN paths
        # SNN expects input with shape [time_steps, batch_size, features]
        # x has shape [batch_size, time_steps, features]
        x_snn = x.permute(1, 0, 2)
        
        # Conventional NN takes the last timestep as input.
        x_conv = x[:, -1, :] 
        
        # SNN forward pass.
        # We process the entire sequence through the SNN model
        snn_output = self.snn_model(x_snn)
        
        # The output of a SNN is generally the sum or mean of spikes over time
        # The output of snn_model has a shape of [time_steps, batch_size, features],
        # we need to average the outputs along the time dimension (dim=0)
        snn_output = snn_output.mean(dim=0)
        
        # Conventional NN forward pass
        conv_output = self.conv_model(x_conv)
        
        # Concatenate outputs from both models along the feature dimension (dim=1)
        fused_output = torch.cat((snn_output, conv_output), dim=1)
        
        # Final classification
        output = self.fusion_layer(fused_output)
        
        return output
