import os
import torch

from src.training.trainer import Trainer
from src.training.compression import quantize_model

def run_training(model, criterion, optimizer, train_loader, val_loader, device, training_config, logger, q_enabled):
    """
    Runs the training loop using the Trainer class
    """
    logger.info('Training final model...')
    qat = q_enabled and training_config['compression_params']['quantization']['method'] == 'qat'

    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device, training_config, qat)
    trainer.run(logger=logger)

    # Save checkpoint
    save_dir = training_config['training_params'].get('model_save_path', 'models')
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_name = f'{getattr(model, 'model_name', 'Model')}.pth'
    prefixes = []

    if q_enabled:
        prefixes.append("Q")
    if training_config['compression_params']['pruning'].get("enabled", False):
        prefixes.append("Pruning")

    checkpoint_name = "_".join(prefixes + [checkpoint_name])
    save_path = os.path.join(save_dir, checkpoint_name)

    torch.save(model.state_dict(), save_path)
    logger.info(f'Model checkpoint saved at {save_path}')

    if q_enabled:
        q_method = training_config['compression_params']['quantization']['method']
        if q_method=='dynamic':
            q_model = quantize_model(model, logger)
            return q_model    
        
    return model