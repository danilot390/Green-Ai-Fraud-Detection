import torch
import numpy as np
import os
import json
import tempfile

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

def _torch_model_size_mb(model: torch.nn.Module) -> float:
    """
    CodeCarbon-aligned model size:
    - CPU serialization
    - state_dict only
    - temporary unique file
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)

    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb

def _xgb_model_size_mb(xgb_model) -> float:
    """
    XGBoost-native serialization (deployment size).
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    xgb_model.save_model(path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb

def get_model_size(model, filename="temp_model.pth"):
    """
    Saves the model state dict to a temporary file and returns its size in MB.
    """
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / (1024*1024)
    os.remove(filename)
    return size_mb

def get_ml_model_size(model, filename="temp_model.json"):
    """
    Saves a scikit-learn model using joblib and returns its size in MB.
    """
    model.save_model(filename)
    size_mb = os.path.getsize(filename) / (1024*1024)
    os.remove(filename)
    return size_mb

def get_ensemble_size(model):
    """
    Saves each model in the ensemble to a temporary file and returns the total size in MB.
    """
    total_size_mb = 0.0
    breakdown = {}

    if hasattr(model, 'cnn_model'):
        breakdown['cnn_model'] = get_model_size(model.cnn_model, filename="temp_cnn.pth")
        total_size_mb += breakdown['cnn_model']

    if hasattr(model, 'lstm_model'):
        breakdown['lstm_model'] = get_model_size(model.lstm_model, filename="temp_lstm.pth")
        total_size_mb += breakdown['lstm_model']

    if hasattr(model, 'transformer_model'):
        breakdown['transformer_model'] = get_model_size(model.transformer_model, filename="temp_transformer.pth")
        total_size_mb += breakdown['transformer_model']

    if hasattr(model, 'meta_model'):
        breakdown['meta_model'] = get_ml_model_size(model.meta_model, filename="temp_meta.json")
        total_size_mb += breakdown['meta_model']

    return total_size_mb, breakdown