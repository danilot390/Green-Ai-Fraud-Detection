import torch
import numpy as np
from src.utils.config_parser import load_config

def get_device(): 
    """ Returns the available device from config file and system capabilities """ 
    training_config = load_config('config/training_config.yaml') 
    device_str = training_config['training_params']['device'] 
    return torch.device( 'cuda' if torch.cuda.is_available() and device_str == 'cuda' 
                        else ('mps' if device_str == 'mps' and torch.backends.mps.is_available() else 'cpu') )

def to_numpy_squeezed(instance):
    """   
    Convert to NumPy arrays and remove extra dimensions
    """
    if isinstance(instance, torch.Tensor):
        instance = instance.detach().numpy()
    instance = np.squeeze(instance)
    return instance
