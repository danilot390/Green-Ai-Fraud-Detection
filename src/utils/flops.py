import torch.nn as nn
import torch
from ptflops import get_model_complexity_info

def _calculate_torch_flops(model, input_shape):
    """
    Returns FLOPs and parameters for a PyTorch model.
    """
    with torch.no_grad():
        flops, _ = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
    return flops

def _calculate_xgboost_flops(xgb_model):
    """
    Approximate inference FLOPs for XGBoost.
    """
    booster = xgb_model.get_booster()
    
    n_trees = len(booster.get_dump())
    avg_depth = xgb_model.max_depth

    flops_per_tree = avg_depth * 2

    total_flops = n_trees * flops_per_tree
    return total_flops


def calculate_flops_hybrid(model, input_tensor, logger, lif_ops=6):
    """
    Manually calculates FLOPs for a Stacked hybrid model.
    """
    total_flops = 0
    # batch_size, time_steps, features = input_tensor.shape

    # Conventional NN Flops
    for layer in model.conv_model.modules():

        if isinstance(layer, nn.Linear):
            # FLOPs for a Linear layer: 2 * in_features * out_features (mult and add)
            total_flops += 2 * layer.in_features * layer.out_features
        elif isinstance(layer, nn.ReLU):
            # FLOPs for ReLU: 1 * number of elements
            total_flops += layer.inplace == False
        elif isinstance(layer, nn.GELU):
            # Approximate GELU FLOPs (tanh-based)
            total_flops += 12 * layer.num_features
        
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
            if prev_out_features is None:
                raise ValueError("LIFNode found before any Linear layer to determine output features.")
            
            total_flops += lif_ops * prev_out_features * snn_time_steps

    # Final Linear Layer (connects SNN and MLP outputs)
    final_linear_layer = model.fusion_layer
    total_flops += 2 * final_linear_layer.in_features * final_linear_layer.out_features

    logger.info(f'Total FLOPs: {total_flops:.4f}')
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
            lstm_out = torch.zeros(
                x.size(0), self.model.lstm.hidden_size * 2
            ).to(x.device)

            xgb_emb = x.view(
                x.size(0), self.model.input_size
            ).to(x.device)

            fusion =  torch.cat([x_cnn, lstm_out, xgb_emb], dim =1)

            return self.model.mlp_stack(fusion)

    # Cnn + MLP
    macs, _ = get_model_complexity_info(
        WrappedModel(model), 
        input_size, 
        as_strings= False, 
        print_per_layer_stat= False,
        verbose= False
        ) 
    
    # MACs -> FLOPs
    macs *= 2

    # LSTM FLOPs
    lstm_flops = estimate_lstm_flops(
        lstm_input_size, 
        lstm_hidden_size, 
        lstm_seq_len)

    total_flops = macs+ lstm_flops
    
    logger.info(f'FLOPs (CNN + MLP): {macs:,}')
    logger.info(f'FLOPs (BiLSTM estimated): {lstm_flops:,}')
    logger.info(f'Total FLOPs: {total_flops:.4f}')
    
    return {
        'FLOPs (CNN + MLP)': macs,
        'FLOPs (BiLSTM estimated)': lstm_flops,
        'Total FLOPs': total_flops
    }

def calculate_flops_ensemble_model(model, input_size, logger):
    """
    Calculate FLOPs for Ileberi-Sun Stacking ensemble model.
    """
    total_flops = 0.0
    breakdown = {}

    if hasattr(model, 'hybrid_model'):
        breakdown['hybrid_model'] = calculate_flops_hybrid(model.hybrid_model, input_size)
        total_flops += breakdown['hybrid_model']

    if hasattr(model, 'cnn_model'):
        breakdown['cnn_model'] = _calculate_torch_flops(model.cnn_model, input_size)
        total_flops += breakdown['cnn_model']

    if hasattr(model, 'lstm_model'):
        breakdown['lstm_model'] = _calculate_torch_flops(model.lstm_model, input_size)
        total_flops += breakdown['lstm_model']

    if hasattr(model, 'transformer_model'):
        breakdown['transformer_model'] = _calculate_torch_flops(model.transformer_model, input_size)
        total_flops += breakdown['transformer_model']

    if hasattr(model, 'meta_model'):
        breakdown['meta_model'] = _calculate_xgboost_flops(model.meta_model)
        total_flops += breakdown['meta_model']

    for k, v in breakdown.items():
        logger.info(f'FLOPs ({k}): {v:,}')

    logger.info(f'Total FLOPs: {total_flops:.4f}')
    
    return total_flops