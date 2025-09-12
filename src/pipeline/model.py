import torch
from torch import nn 
from torch.optim import Adam

from src.utils.model_utils import load_model
from src.training.compression import prepare_qat_model, convert_qat_model

def setup_model(input_size, time_steps, device, model_config, training_config, y_train, logger):
    """
    Model loading plus compression techniques(QAT/ pruning)
    """
    model = load_model(input_size, time_steps, device, logger, model_config)

    # Is Hybrid
    is_hybrid = model.model_name == 'HybridModel'

    # Compression techniques
    qat_enabled = training_config['compression_params']['quantization'].get('enabled', False)
    pruning_enabled = training_config['compression_params']['pruning'].get('enabled', False)

    if qat_enabled:
        model = prepare_qat_model(model, training_config['compression_params']['quantization'], logger)

    # Weighted loss
    num_pos = (y_train == 1 ).sum().item()
    num_neg = (y_train == 0 ).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=training_config['training_params']['learning_rate'])

    logger.info(f'Model selected: {model.model_name}')
    logger.info(f'Using device: {device}')
    if qat_enabled or pruning_enabled:
        logger.info(f'Compression techniques enabled:')
        logger.info(f'\t * Quantization-Aware Train.') if qat_enabled else None
        logger.info(f'\t * Pruning') if pruning_enabled else None

    return model, criterion, optimizer, qat_enabled, is_hybrid

def finalize_model(model, qat_enabled, model_config, logger):
    """
    Convert to a quantized model if is require.
    """
    if qat_enabled:
        model = convert_qat_model(model, model_config['compression_params']['quantization'], logger)

    return model
