import torch
import os
import numpy as np 

from src.models.conventional_model import ConventionalNN
from src.models.hybrid_model import HybridModel
from src.models.snn_model import SNNModel

def get_model_use(model_config):
    """
    Get active models based on model config.
    """
    # Check which model use
    is_snn_model = model_config['snn_model']['enabled']
    is_conventional_model = model_config['conventional_nn_model']['enabled']
    is_hybrid_model = is_snn_model and is_conventional_model

    return {
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

