import torch.nn as nn

def calculate_flops(model, input_tensor):
    """
    Manually calculates FLOPs for the hybrid model.
    """
    total_flops = 0
    
    # MLP FLOPs
    mlp_input = input_tensor[:, -1, :]  # Take the last time step for MLP
    for layer in model.conv_model.children():
        if isinstance(layer, nn.Linear):
            # FLOPs for a Linear layer: 2 * in_features * out_features (mult and add)
            total_flops += 2 * layer.in_features * layer.out_features
        
    # SNN FLOPs
    time_steps = input_tensor.shape[1]
    snn_input = input_tensor
    for name, module in model.snn_model.named_children():
        if "dense" in name: # Linear layers in SNN
            # FLOPs for SNN linear layers: 2 * in_features * out_features * time_steps
            # Multiplied by time_steps because operations are performed at each step
            in_features = module.in_features
            out_features = module.out_features
            total_flops += 2 * in_features * out_features * time_steps
        
        # LIFNode operations are simpler, mostly additions for the membrane potential.
        # This is a good place for an estimation. We'll count a few ops per neuron per time step.
        elif "LIFNode" in module.__class__.__name__:
            # A simplified estimation for LIF: 2 ops per neuron per time step (decay & update)
            num_neurons = module.v.shape[0] if len(module.v.shape) > 1 else module.v.shape[0]
            total_flops += 2 * num_neurons * time_steps

    # Final Linear Layer (connects SNN and MLP outputs)
    final_linear_layer = model.fusion_layer
    total_flops += 2 * final_linear_layer.in_features * final_linear_layer.out_features

    return total_flops
