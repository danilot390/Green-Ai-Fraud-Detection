import torch
from torch import nn 
from torch.optim import Adam

from src.utils.model_utils import load_model
from src.training.compression import prepare_qat_model, convert_qat_model
from src.models.losses import FocalLoss

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
    loss_function = training_config['training_params']['loss_config']
    if  loss_function['type'] =='FocalLoss':
        # Focal Loss
        criterion = FocalLoss(alpha=loss_function.get('alpha',1.0), gamma=loss_function.get('gamma',1.5))
    elif loss_function['type'] =='BCEWithLogitsLoss':
        # Weighted loss
        num_pos = (y_train == 1 ).sum().item()
        num_neg = (y_train == 0 ).sum().item()
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        logger.error('Loss function selected is not well configured.')

    logger.info(f'Loss Function for criterion: {loss_function}')
    optimizer = Adam(model.parameters(), lr=training_config['training_params']['learning_rate'])

    logger.info(f'Model selected: {model.model_name}')
    logger.info(f'Using device: {device}')
    if qat_enabled or pruning_enabled:
        logger.info(f'Compression techniques enabled:')
        logger.info(f'\t * Quantization-Aware Train.') if qat_enabled else None
        logger.info(f'\t * Pruning') if pruning_enabled else None

    return model, criterion, optimizer, qat_enabled, is_hybrid

def finalize_model(model, qat_enabled, training_config, logger):
    """
    Convert to a quantized model if is require.
    """
    if qat_enabled:
        model = convert_qat_model(model, training_config['compression_params']['quantization'], logger)

    return model
