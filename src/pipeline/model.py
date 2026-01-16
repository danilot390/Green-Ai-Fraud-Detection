import torch
from torch import nn 
from torch.optim import Adam

from src.pipeline.utils import get_optimizer, build_critetrion
from src.pipeline.model_factory import load_model
from src.training.compression import convert_qat_model

def setup_model(input_size, time_steps, device, model_config, training_config, y_train, logger):
    """
    Model loading plus compression techniques(QAT/ pruning)
    """
    l_model = load_model(input_size, time_steps, device, logger, model_config)
    
    # Compression techniques
    qat_enabled = training_config['compression_params']['quantization'].get('enabled', False)
    pruning_enabled = training_config['compression_params']['pruning'].get('enabled', False)
    training_params = training_config['training_params']
    logger.info(f'Using device: {device}')
    if qat_enabled or pruning_enabled:
        logger.info(f'Compression techniques enabled:')
        logger.info(f'\t * Quantization-Aware Train.') if qat_enabled else None
        logger.info(f'\t * Pruning') if pruning_enabled else None

    criterion = build_critetrion(training_params, y_train, device, logger)
    if isinstance(l_model, nn.Module):
        model = l_model
        model.to(device)

        is_hybrid = model.model_name == 'HybridModel'
        is_ensemble = False

        optimizer = get_optimizer(model, training_params)

        logger.info(f'Model selected: {model.model_name}')
        return model, criterion, optimizer, qat_enabled, is_hybrid, is_ensemble
    ensemble_model = None
    base_models = None

    if not isinstance(l_model, dict):
        raise TypeError(
            f"Expected 'l_model' to be dict, got {type(l_model).__name__}"
        )
    if "ensemble" not in l_model:
        raise KeyError(
            "Ensemble dict must contain key 'ensemble'. "
        )
    ensemble_model = l_model["ensemble"]
    base_models = {k: v for k, v in l_model.items() if k != "ensemble"}


    optimizers = {
        name: get_optimizer(m, training_params) for name, m in base_models.items()
    }

    is_hybrid, is_ensemble = False, True

    logger.info(f"Ensemble selected: {type(ensemble_model).__name__}")

    return ensemble_model, base_models, criterion, optimizers, qat_enabled, is_hybrid, is_ensemble

def finalize_model(model, qat_enabled, training_config, logger):
    """
    Convert to a quantized model if is require.
    """
    if qat_enabled:
        model = convert_qat_model(model, training_config['compression_params']['quantization'], logger)

    return model
