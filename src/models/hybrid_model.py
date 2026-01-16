import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron

class HybridModel(nn.Module):
    def __init__(self, snn_input_size, snn_time_steps, config):
        """
        A hybrid model combining a Spiking Neural Network (SNN) and a Conventional Neural Network (CNN).
        The SNN and CNN operate in parallel on the same input, and their outputs are for a fused
        final classification or either stacked for further processing.
        """
        super().__init__()
        self.snn_time_steps = snn_time_steps
        self.meta_learning = config['hybrid_model']['meta_learning'].get('enabled', False)
        self.model_name = type(self).__name__ 
        
        # Get the hidden size from config
        snn_config = config['snn_model']
        conv_config = config['conventional_nn_model']
        snn_hidden_size = snn_config['hidden_layers'][-1]['units']
        conv_hidden_size = conv_config['mlp_layers'][-1]['units']

        self.snn_hidden_size = snn_hidden_size
        self.conv_hidden_size = conv_hidden_size
        
        # SNN part: acts as one of the feature extractors
        snn_layers = []
        prev_size = snn_input_size

        for layer_params in snn_config['hidden_layers']:
            units = layer_params['units']
            tau = layer_params.get('tau', 2.0)
            beta = layer_params.get('beta', 0.95)
            dropout = layer_params.get('dropout', 0.0)

            snn_layers.append(
                nn.Linear(prev_size, units)
            )
            # BatchNorm
            if layer_params.get("batchnorm", False):
                snn_layers.append(
                    nn.LayerNorm(units)
                )
            # LIF Node
            snn_layers.append(
                neuron.LIFNode(
                    tau=tau,
                    decay_input=beta)
            )
            # Dropout
            if dropout > 0.0:
                snn_layers.append(
                    nn.Dropout(dropout)
                )
            prev_size = units

        self.snn_model = nn.Sequential(*snn_layers)
        
        # Conventional NN part: acts as the other feature extractor
        mlp_layers = []
        input_dim = snn_input_size

        for layer_params in conv_config['mlp_layers']:
            units = layer_params['units']
            dropout_rate = layer_params.get('dropout_rate', 0.0)
            batchnorm = layer_params.get('batchnorm', True)

            mlp_layers.append(nn.Linear(input_dim, units))

            # Batchnorm
            if batchnorm:
                mlp_layers.append(nn.BatchNorm1d(units))

            mlp_layers.append(nn.ReLU())

            # Dropout
            if dropout_rate > 0:
                mlp_layers.append(nn.Dropout(dropout_rate))

            input_dim = units  # update for next loop
        conv_hidden_size = input_dim
        self.conv_model = nn.Sequential(*mlp_layers)
        
        # Fusion layer and final output
        # The input size is the sum of the hidden sizes from both models
        self.fusion_layer = nn.Linear(snn_hidden_size + conv_hidden_size, 1)

    def extract_fused_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the fused feature representation from the SNN and MLP branches.
        """
        # Temporal dimension
        if x.dim() == 2:
            # [B, F] -> [B, 1, F]
            x = x.unsqueeze(1)

        # Reset the SNN state
        functional.reset_net(self.snn_model)
        
        # SNN branch: expects [time_steps, batch_size, features]
        x_snn = x.permute(1, 0, 2)
        snn_output = self.snn_model(x_snn)  # [time_steps, batch_size, snn_hidden_size]
        snn_output = snn_output.mean(dim=0) # [batch_size, snn_hidden_size]
        
        # MLP branch: last time-step
        x_conv = x[:, -1, :]                 # [batch_size, features]
        conv_output = self.conv_model(x_conv) # [batch_size, conv_hidden_size]
        
        # Fused features
        fused_output = torch.cat((snn_output, conv_output), dim=1)
        return fused_output

    def forward(self, x):
        """
        Performs the forward pass for the hybrid model.
        """
        fused_output = self.extract_fused_features(x)
        logits = self.fusion_layer(fused_output)
        return logits.squeeze(1) 
