import torch
import numpy as np
import os
import json

def get_device(device='cpu'): 
    """ Returns the available device from config file and system capabilities """ 
    return torch.device( 'cuda' if torch.cuda.is_available() and device == 'cuda' 
                        else ('mps' if device == 'mps' and torch.backends.mps.is_available() else 'cpu') )

def to_int_array(instance):
    """
    Convert input to a 1D integer Numpy array, supports CPU/GPU tensors.
    """
    if isinstance(instance, torch.Tensor):
        instance = instance.detach().numpy()
    instance = np.squeeze(np.asarray(instance)).astype(int)

    return instance

def save_to_json(dict, dir, file, logger):
    """
    Save dictionary into a JSON file based on parameters.
    """
    data = {}
    for key, value in dict.items():
        data[key] = value

    with open(os.path.join(dir,file), 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f'The `{file}` is saved in `{dir}`.')

    return
