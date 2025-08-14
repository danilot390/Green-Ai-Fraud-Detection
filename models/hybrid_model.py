import torch
import torch.nn as nn
from models.snn_model import SNNModel
from models.conventional_model import ConventionalNN

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # SNN acts as the feature extractor
        self.snn_extractor = SNNModel(
            input_size = self.config['snn_model']['input_size'],
            config = self.config,
            time_steps = self.config['snn_model']['time_steps'],
            hybrid=True,
        )

        # the SNN's output size => CNN's input_size  
        conventional_nn_input_size = self.config['snn_model']['hidden_layers'][-1]['units']

        print('snn output.shape: {snn_output_size.shape}')
        print('config.shape{self.company.shape}')
        #  The CNN performs the final classification
        self.conventional_classifier = ConventionalNN(
            input_size= conventional_nn_input_size,
            config = self.config
        )

    def forward(self, x):
        # SNN forward pass with sequential data
        snn_output = self.snn_extractor(x)

        # SNN's ouput => CNN
        final_output = self.conventional_classifier(snn_output)

        return final_output
    