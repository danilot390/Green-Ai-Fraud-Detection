import torch
import torch.nn as nn
import os
from torch.optim import Adam, SGD, Adagrad

from src.models.losses import FocalLoss

def get_optimizer(model: nn.Module, training_params):
    lr = training_params["learning_rate"]
    optimizer_name = training_params.get('optimizer', 'adam')

    optimizers = {
        'adam': Adam,
        'sgd': SGD,
        'adagrad': Adagrad,
    }

    optimizer_select = optimizers.get(optimizer_name, Adam)

    if optimizer_name.lower() not in optimizers:
        print(f'Unsupported optimizer: {optimizer_name}')
        print('Defaulting to Adam optimizer.')

    return optimizer_select(model.parameters(), lr=lr)


def build_critetrion(trainng_config, y_train, device, logger):
    loss_cfg = trainng_config.get('loss_config', 'BCEWithLogitsLoss')
    
    if loss_cfg["type"] == "FocalLoss":
        alpha=loss_cfg.get("alpha", 1.0)
        gamma=loss_cfg.get("gamma", 1.5)
        logger.info(f'Loss Function for criterion: {loss_cfg['type']}; alpha: {alpha}, gamma: {gamma}')
        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
        )
    elif loss_cfg["type"] == "BCEWithLogitsLoss":
        # Weighted loss
        num_pos = (y_train == 1).sum().item()
        num_neg = (y_train == 0).sum().item()
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device, dtype=torch.float)
        logger.info(f'Loss Function for criterion: {loss_cfg['type']}; pos-weight: {pos_weight.item():.4f}')
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logger.error("Loss function selected is not well configured.")
    raise ValueError(f"Unsupported loss_config: {loss_cfg}")

def get_best_model(model, name_metric, metric, epoch, best_val_metric, best_epoch, no_improve_epochs, model_save_path, model_name,cfg, logger):    
    """
    Saves the best model based whe a new metric is achieved, or meets pruning/quantization conditions allow tolereance-based updates.
    """
    save_model = qat = False
    pruning_config, quant_config = cfg.get("pruning", {}), cfg.get("quantization", {})

    if metric > best_val_metric:
        best_val_metric = metric
        best_epoch = epoch
        save_model = True
    else:
        tolerance = 0.001
        # Check pruning & quantization conditions
        if pruning_config.get("enabled", False) and pruning_config.get('start_epoch',3)<epoch:
            tolerance = pruning_config.get("tolerance", tolerance)
        elif quant_config.get("enabled", False) and quant_config.get('start_epoch',5)<epoch:
            tolerance, qat = quant_config.get("tolerance", tolerance), True

        if metric >= best_val_metric - tolerance:
            save_model = True
    
    if save_model:
        no_improve_epochs = 0
        torch.save({
            'model_state_dict':model.state_dict(),
            'epoch': epoch,
            'metric_score': best_val_metric,
            'qat_enabled': qat
        }, os.path.join(model_save_path, model_name))
        logger.info(f"New best model saved with {name_metric}: {best_val_metric:.4f}")
    else:
        no_improve_epochs += 1

    return best_val_metric, best_epoch, no_improve_epochs
