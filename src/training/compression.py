import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization as tq


#  Quantization (Dynamic)
def quantize_model(model, logger):
    logger.info(" Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    logger.info(" Model quantized (dynamic).")
    return quantized_model

# Quantization-Aware Training (QAT)
def prepare_qat_model(model, quant_config, logger, qat_start_epoch, current_epoch):
    """
    Prepare the model for Quantization-Aware Training (QAT).
    Moves model to CPU (required), applies observer configuration, and prepares QAT modules.
    """

    if current_epoch < qat_start_epoch:
        return model
    
    def attach_qconfig(model):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                module.qconfig = qconfig
            else:
                attach_qconfig(module)

    logger.info(f'Quantization-Aware Training (QAT) activated at epoch {current_epoch}')

    if quant_config.get('quantization_scheme', 'symmetric').lower() == 'symmetric':
        qscheme = torch.per_tensor_symmetric  
    else: 
        qscheme= torch.per_tensor_affine
    dtype = torch.qint8
    
    # Observer selection
    observer_type = quant_config.get('observer_type', 'minmax')
    if observer_type == 'minmax':
        activation_observer = tq.MinMaxObserver.with_args(dtype=dtype, qscheme=qscheme, quant_min=-128, quant_max=127)
    elif observer_type == 'moving_average_minmax':
        activation_observer = tq.MovingAverageMinMaxObserver.with_args(dtype=dtype, qscheme=qscheme, quant_min=-128, quant_max=127)
    else:
        logger.error(f"Unsupported observer type: {observer_type}")
        raise ValueError(f"Unsupported observer type: {observer_type}")

    weight_observer = tq.PerChannelMinMaxObserver.with_args(
            dtype=dtype,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            quant_min=-128,
            quant_max=127
        )
    
    qconfig = tq.QConfig(
        activation= activation_observer,
        weight= weight_observer,
    )
    logger.info(f"Using observer: {observer_type}, qscheme: {qscheme}")

    # QAT requires CPU
    device = next(model.parameters()).device
    if device.type != 'cpu':
        logger.warning("Model must be on CPU for QAT preparation. Moving model to CPU.")
        model = model.to('cpu')

    attach_qconfig(model)
    
    model.train()
    # Validate all weights are nn.Parameter
    for name, param in model.named_parameters():
        if not isinstance(param, nn.Parameter):
            logger.error(f"Parameter {name} is not a torch.nn.Parameter — invalid assignment detected.")
            raise TypeError(f"Parameter {name} is not a torch.nn.Parameter — invalid assignment detected.")
    
    model = tq.prepare_qat(model, inplace=False)
    logger.info("Model is prepared for QAT.")
    return model


def convert_qat_model(model, quant_config, val_loader,logger):
    """
    Converts the QAT-trained model to a quantized version.
    """
    if not quant_config.get("enabled", False):
        return model

    logger.info("Converting QAT model to quantized version...")
    model.eval()

    with torch.no_grad():
        # Calibration run (on val set or dummy data)
        for i, batch in enumerate(val_loader):
            x = batch[0]
            _ = model(x)
            if i >= 10:
                break

    model = tq.convert(model, inplace=False)
    logger.info("QAT model successfully converted to quantized model.")
    return model

# Pruning
def apply_pruning(model, epoch, pruning_config, logger):
    """
    Applies pruning to the model according to the configuration and schedule.
    Supports unstructured and structured pruning.
    """
    if not pruning_config.get('enabled', False):
        return model
    
    method = pruning_config.get('method', 'l1_unstructured')
    base_amount = pruning_config.get('amount', 0.2)
    schedule = pruning_config.get('schedule', 'one_shot')
    start_epoch = pruning_config.get('start_epoch', 0)
    end_epoch = pruning_config.get('end_epoch', 0)
    frequency = pruning_config.get('frequency', 1)

    if epoch < start_epoch or (end_epoch and epoch > end_epoch):
        return model
    logger.info(f'Pruning technique started...')

    # Compute pruning
    amount = base_amount
    if schedule == 'gradual':
        progress = (epoch - start_epoch) / max(1, (end_epoch - start_epoch))
        amount = min(progress * base_amount, 1.0)  # Clamp to 1.0

    if (epoch - start_epoch) % frequency == 0:
        pruned = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=amount)
                elif method == 'ln_structured':
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                else:
                    logger.error(f"Unsupported pruning method: {method}")
                    raise ValueError(f"Unsupported pruning method: {method}")
                pruned += 1
        if amount > 0.0:
            logger.info(f"Pruned {pruned} layers using method: {method}, amount: {amount:.4f} at epoch {epoch}")

    return model


def remove_pruning(model, logger):
    """
    Permanently removes pruning reparameterizations (makes sparsity permanent).
    """
    pruned = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
            pruned += 1
    if pruned >0 :
        logger.info(f"Removed pruning from {pruned} layers. Model is now sparse.")
    return model
