import torch
import torch.nn as nn
from torch.nn.utils import prune
import torch.ao.quantization as tq

def prepare_qat_model(model, quant_config,logger):
    """
    Preapare model for QAT based on config
    """
    if not quant_config.get('enabled', False):
        return model

    logger.info('Preparing model for Quantization-Aware Train (QAT)...')

    # Observer type
    if quant_config['observer_type'] == 'minmax':
        activation_observer = tq.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric 
            if quant_config['quantization_scheme'] == 'symmetric' 
            else torch.per_tensor_affine,
        )
    elif quant_config['observer_type'] == 'moving_average_minmax':
        activation_observer = tq.MovingAverageMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric 
            if quant_config['quantization_scheme'] == 'symmetric' 
            else torch.per_tensor_affine
        )
    else:
        logger.error(f'Unsupported oobserver type: ´{quant_config['observer_type']}´')
        raise ValueError(f'Unsupported observer type `{quant_config['observer_type']}`')
    
    # QAT Config
    qconfig = tq.QConfig(
        activation = activation_observer,
        weight=tq.default_weight_observer
    )

    model.qconfig = qconfig
    logger.warning('Model must be on CPu for QAT preaparation.')
    model = model.to('cpu')
    model = tq.prepare_qat(model.train(), inplace=False)

    logger.info('Model prepared for QAT (with fake quantization). Continue training...')
    return model

def convert_qat_model(model, quant_config, logger):
    """
    Convert trained QAT model to quantized model
    """
    if not quant_config.get("enabled", False):
        return model

    logger.info("Converting QAT model to quantized version...")
    model = tq.convert(model.eval(), inplace=False)
    logger.info("QAT model converted successfully.")
    return model

def apply_pruning(model, epoch, pruning_config, logger):
    """
    Apply pruning on model based on settings.
    """
    if not pruning_config.get('enabled', False):
        return model
    
    method = pruning_config.get('method', 'l1_unstructured')
    amount = pruning_config.get('amount', 0.2)
    schedule = pruning_config.get('schedule', 'one_shot')
    start_epoch = pruning_config.get('start_epoch', 0)
    end_epoch = pruning_config.get('end_epoch', 0)
    frequency = pruning_config.get('frequency', 1)

    if epoch < start_epoch or (end_epoch and epoch >end_epoch):
        return model
    
    if schedule == 'gradual':
        progress = (epoch -start_epoch)/ max(1, (end_epoch - start_epoch))
        amount = progress *amount

    if (epoch -start_epoch)% frequency == 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount= amount)
                elif method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=amount)
                elif method == 'ln_structured':
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                else:
                    logger.error(f'Unsupported pruning method: {method}')
                    raise ValueError(f'Unsupported pruning method: {method}')

        logger.info(f'Applied {schedule} pruning at epoch {epoch}.')

    return model

def remove_pruning(model, logger):
    """
    Sparse the model by removing pruning reparametrization. 
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear)and hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
    logger.info('Removed pruning reparametrization. Model is now sparse.')
    return model


    