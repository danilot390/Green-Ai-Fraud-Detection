import torch.nn as nn
import torch
from ptflops import  get_model_complexity_info

def calculate_flops_hybrid(model, input_tensor):
    """
    Manually calculates FLOPs for the hybrid model.
    """
    total_flops = 0
    batch_size, time_steps, features = input_tensor.shape

    # Conventional NN Flops
    for layer in model.conv_model.children():
        if isinstance(layer, nn.Linear):
            # FLOPs for a Linear layer: 2 * in_features * out_features (mult and add)
            total_flops += 2 * layer.in_features * layer.out_features
        
    # SNN FLOPs
    snn_time_steps = model.snn_time_steps
    prev_out_features = None

    for module in model.snn_model:
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            prev_out_features = out_features
            total_flops += 2* in_features* out_features* snn_time_steps
        
        elif "LIFNode" in module.__class__.__name__ and isinstance(module, nn.Module):
            # A simplified estimation for LIF: 2 ops per neuron per time step (decay & update)
            total_flops += 2 * prev_out_features * snn_time_steps

    # Final Linear Layer (connects SNN and MLP outputs)
    final_linear_layer = model.fusion_layer
    total_flops += 2 * final_linear_layer.in_features * final_linear_layer.out_features

    return total_flops

def estimate_lstm_flops(input_size, hidden_size, seq_len, num_directions=2):
    """
    Approx FLOPs for an LSTM layer
    """
    lstm_flops = 8* hidden_size* (hidden_size+ input_size)* seq_len* num_directions
    return lstm_flops

def calculate_flops_hybrid_ml(model, input_size, lstm_input_size, lstm_hidden_size, lstm_seq_len, logger):
    """
    Calculate FLOPs and parameters for XGBoost + CNN + BiLSTM
    """
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model 
        
        def forward(self, x):
            # Cnn + MLP
            x_cnn =self.model.cnn_stack(x)
            x_cnn = x_cnn.view(x.size(0), -1)

            # fake LSM output
            lstm_out = torch.zeros(x.size(0), self.model.lstm.hidden_size * 2).to(x.device)

            xgb_emb = x.view(x.size(0), -1)

            fusion =  torch.cat([x_cnn, lstm_out, xgb_emb], dim =1)

            return self.model.mlp_stack(fusion)

    # Cnn + MLP
    macs, _ = get_model_complexity_info(WrappedModel(model), input_size, as_strings= False, print_per_layer_stat= False ) 
    # LSTM FLOPs
    lstm_flops = estimate_lstm_flops(lstm_input_size, lstm_hidden_size, lstm_seq_len)

    total_flops = macs+ lstm_flops
    
    logger.info(f'FLOPs (CNN + MLP): {macs:,}')
    logger.info(f'FLOPs (BiLSTM estimated): {lstm_flops:,}')
    logger.info(f'Total FLOPs: {total_flops:.4f}')
    
    return {
        'FLOPs (CNN + MLP)': {macs},
        'FLOPs (BiLSTM estimated)': {lstm_flops},
        'Total FLOPs': {total_flops}
    }