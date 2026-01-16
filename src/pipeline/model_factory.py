import torch.nn as nn

from src.models.registry import MODEL_REGISTRY
from src.ensemble.registry import ENSEMBLE_REGISTRY

def load_model(input_size, time_steps, device, logger, model_config):
    """
    Load model based on model config.
    """
    # Check which model use
    model_cls = get_model_use(model_config)

    if model_cls.__name__ in ['HybridModel', 'SNNModel', 'snn']:
        model = model_cls(input_size, time_steps, model_config).to(device)
    elif model_cls.__name__ == 'ConventionalNN':
        model = model_cls(input_size=input_size, config=model_config).to(device)
    elif model_cls.__name__ ==  'XBoost_CNN_BiLSTM':
        config = model_config['xgb_cnn_bilstm_model']
        model = model_cls(
            input_size=input_size, 
            cnn_config=config['cnn_config'], 
            lstm_hidden=config['lstm_config'].get('hidden_size', 64), 
            mlp_config=config['mlp_config'])
    elif model_cls.__name__ == 'IleberiSunStackingModel':
        model = i_s_setup_model(input_size, device, model_config, logger)
    else:
        raise ValueError(f'Unsupported model class: {model_cls.__name__}')
    if isinstance(model, nn.Module):
        model = model.to(device)
    logger.info("Model %s loaded successfully.", model_cls.__name__)
    return model

def get_model_use(model_config):
    """
    Return the selcted model class based on config.
    """
    # Check which model use
    model_flags = [
        ('hybrid', model_config['hybrid_model']),
        ('ileberi_sun', model_config['Ileberi_Sun_model']),
        ('xboost_cnn_bilstm', model_config['xgb_cnn_bilstm_model']),
        ('snn', model_config['snn_model']),
        ('conventional', model_config['conventional_nn_model']),   
    ]

    for key, config in model_flags:
        if config.get('enabled', False):
            if key in MODEL_REGISTRY:
                return MODEL_REGISTRY[key]
            if key in ENSEMBLE_REGISTRY:
                return ENSEMBLE_REGISTRY[key]
            
    raise ValueError('Invalid model configuration: no valid model enabled.')

def i_s_setup_model(input_size, device, model_config, logger):
    """
    Specific setup for Ileberi-Sun Stacking model.
    """
    cnn = MODEL_REGISTRY['ileberi_cnn'](input_size).to(device)
    lstm = MODEL_REGISTRY['lstm'](input_size=input_size).to(device)
    transformer_cfg = model_config.get('transformer_model', {})
    transformer = MODEL_REGISTRY['transformer'](input_size, transformer_cfg).to(device)
    
    xgb = model_config.get('xgboost_params', None)
    ensemble = ENSEMBLE_REGISTRY['ileberi_sun']
    model = ensemble(
        cnn_model=cnn, 
        lstm_model=lstm, 
        transformer_model=transformer, 
        device=device,
        xgb_params=xgb,
        )

    return {
        'ensemble': model,
        'cnn': cnn,
        'lstm': lstm,
        'transformer': transformer,
    }
