import os
import torch

from src.training.trainer import Trainer

def run_training(model, criterion, optimizer, input_size, time_steps, train_loader, val_loader, device, training_config, logger, qat_enabled):
    """
    Runs the training loop using the Trainer class
    """
    logger.info('Training final model...')
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device, training_config, qat_enabled)
    trainer.run(logger=logger)

    # Save checkpoint
    save_dir = training_config['training_params'].get('model_save_path', 'models')
    os.makedirs(save_dir, exist_ok=True)

    checkpoiont_name = f'final_{'qat_'if qat_enabled else''}{model.model_name}.pth'
    save_path = os.path.join(save_dir, checkpoiont_name)

    torch.save(model.state_dict(), save_path)
    logger.info(f'Model checkpoint saved at {save_path}')

    return model