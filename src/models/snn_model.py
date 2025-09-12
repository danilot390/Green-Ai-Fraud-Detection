import torch
import torch.nn as nn 
import snntorch as snn 
from snntorch import surrogate
from snntorch import functional as SF 

class SNNModel(nn.Module):
    def __init__(self, input_size, time_steps, config, hybrid=False):
        super().__init__()
        self.config = config['snn_model']
        self.input_size = input_size
        self.time_steps = time_steps
        self.hybrid = hybrid
        self.model_name = type(self).__name__ 

        # Define the surrogate gradient for backpropagation
        spike_grad = surrogate.fast_sigmoid()
        
        # Network Layers 
        # 1st Standard layer
        self.fc1 = nn.Linear(self.input_size, self.config['hidden_layers'][0]['units'])
        # 1st snn Layer - LIF neuron
        self.lif1 = snn.Leaky(beta=self.config['hidden_layers'][0]['beta'], spike_grad=spike_grad)

        # 2nd Layer
        self.fc2 = nn.Linear(self.config['hidden_layers'][0]['units'], self.config['hidden_layers'][-1]['units'])
        self.lif2 = snn.Leaky(beta=self.config['hidden_layers'][-1]['beta'], spike_grad=spike_grad)

        # Output Layer
        self.fc3 = nn.Linear(self.config['hidden_layers'][-1]['units'], 1)

    def forward(self, data):
        # Membrane for the LIF neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Output list 
        spk_out = []
        mem_out = []
        mem_out_final_layer = []

        _, num_timesteps, _ = data.shape

        # Fowards pass now runs over a numbmer of time
        for step in range(num_timesteps):
            x = data[:, step,:] if self.hybrid else data

            # layer 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem_out_final_layer.append(mem2)

            # Output Layer
            cur3 = self.fc3(spk2)

            # Store spikes & output
            spk_out.append(spk2)
            mem_out.append(cur3)

        snn_features = torch.stack(mem_out_final_layer, dim=0).mean(dim=0)
        final_output = torch.stack(mem_out).mean(dim=0)

        return snn_features if self.hybrid else final_output
    
    def reset_membranes(self):
        self.lif1.init_leaky()
        self.lif2.init_leaky()

        return