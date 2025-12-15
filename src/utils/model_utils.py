import torch
import os
import numpy as np 

from src.models.conventional_model import ConventionalNN
from src.models.hybrid_model import HybridModel
from src.models.snn_model import SNNModel
from src.models.xbost_cnn_bilstm import XBoost_CNN_BiLSTM

def get_model_use(model_config):
    """
    Get active models based on model config.
    """
    # Check which model use
    is_ml_cnn_bl_model = model_config['xgb_cnn_bilstm_model'].get('enabled', False)
    is_snn_model = model_config['snn_model'].get('enabled', False)
    is_conventional_model = model_config['conventional_nn_model'].get('enabled', False)
    is_hybrid_model = model_config['hybrid_model'].get('enabled', False)

    return {
        "ml_cnn_bl_model": is_ml_cnn_bl_model,
        "snn": is_snn_model,
        "conventional": is_conventional_model,
        "hybrid": is_hybrid_model
    }

def load_model(input_size, time_steps, device, logger, model_config):
    """
    Load model based on model config.
    """
    # Check which model use
    model_flags = get_model_use(model_config)

    if model_flags['hybrid']:
        model = HybridModel(snn_input_size=input_size, snn_time_steps=time_steps, config=model_config).to(device)
    elif model_flags['snn']:
        model = SNNModel(input_size=input_size, time_steps=time_steps, config=model_config).to(device)
    elif model_flags['conventional']:
        model = ConventionalNN(input_size=input_size, config=model_config).to(device)
    elif model_flags['ml_cnn_bl_model']:
        cnn_config = model_config['xgb_cnn_bilstm_model']['cnn_config']
        mlp_config = model_config['xgb_cnn_bilstm_model']['mlp_config']
        lstm_config = model_config['xgb_cnn_bilstm_model']['lstm_config']
        model = XBoost_CNN_BiLSTM(input_size=input_size, cnn_config=cnn_config, lstm_hidden=lstm_config.get('hidden_size', 64), mlp_config=mlp_config)
    else:
        logger.error("Invalid model configuration. At least one model must be enabled.")
        raise ValueError("Invalid model configuration. At least one model must be enabled.")
    logger.info(f'Load model successfully...')
    return model

def get_model_size(model, filename="temp_model.pth"):
    """
    Saves the model state dict to a temporary file and returns its size in MB.
    """
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024*1024)
    os.remove(filename)
    return size_mb

def get_best_model(model, f2, epoch, best_val_f2, best_epoch, no_improve_epochs, model_save_path, model_name,cfg, logger):
    """
    Saves the best model based whe a new f1 score is achieved, or meets pruning/quantization conditions allow tolereance-based updates.
    """
    save_model = qat = False
    pruning_config, quant_config = cfg.get("pruning", {}), cfg.get("quantization", {})

    if f2 > best_val_f2:
        best_val_f2 = f2
        best_epoch = epoch
        save_model = True
    else:
        tolerance = 0.001
        # Check pruning & quantization conditions
        if pruning_config.get("enabled", False) and pruning_config.get('start_epoch',3)<epoch:
            tolerance = pruning_config.get("tolerance", tolerance)
        elif quant_config.get("enabled", False) and quant_config.get('start_epoch',5)<epoch:
            tolerance, qat = quant_config.get("tolerance", tolerance), True

        if f2 >= best_val_f2 - tolerance:
            save_model = True
    
    if save_model:
        no_improve_epochs = 0
        torch.save({
            'model_state_dict':model.state_dict(),
            'epoch': epoch,
            'f2_score': best_val_f2,
            'qat_enabled': qat
        }, os.path.join(model_save_path, model_name))
        logger.info(f"New best model saved with F2-score: {best_val_f2:.4f}")
    else:
        no_improve_epochs += 1

    return best_val_f2, best_epoch, no_improve_epochs
