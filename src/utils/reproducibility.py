import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for reproducibility across different libraries.
    """
    print(f"Setting random seed to {seed}...")
    
    # Python and NumPy seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    # Deterministic mode for cuDNN (NVIDIA GPUs only)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False