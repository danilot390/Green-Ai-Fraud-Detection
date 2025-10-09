import torch
import torch.nn as nn
from torch.optim import Adam
import os

from src.training.compression import prepare_qat_model, convert_qat_model, apply_pruning, remove_pruning
from src.utils.config_parser import load_config
from src.utils.common import get_device
from src.utils.model_utils import load_model, get_best_model
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger
from src.data.dataloaders import get_dataloaders
from src.pipeline.setup import setup_experiment
class Trainer:
    """
    Handles the full training process for different types of nodels, including Conventional Neural Networks (NNs)
    Spiking Neural Networks (SNNs) and Hybrid models that combine both.

    Class manages the training loop, validation, and support compress techniques including pruning & quantization-
    aware training (QAT), also saveing the best-performing model by F1 Score.
    """
    def __init__(self, model,criterion, optimizer, train_loader, val_loader, device, config, qat=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.training_config = config
        self.qat = qat

    def train_epoch(self):
        """
        Runs one full epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        # Reset the SNN's state before each epoch
        if hasattr(self.model, 'reset_membranes'):
            self.model.reset_membranes()

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            # Reshape target to match hte model's output shape
            if self.model.model_name == 'HybridModel':
                target = target[:, -1, :].reshape(output.shape)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Runs one full epoch of validation.
        """
        self.model.eval()
        total_loss = 0
        all_labels, all_predictions = [], []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Reset the SNN's state for each batch
                if hasattr(self.model, 'reset_membranes'):
                    self.model.reset_membranes()

                output = self.model(data)

                # Reshape target to match hte model's output shape
                if self.model.model_name == 'HybridModel':
                    target = target[:, -1, :].reshape(output.shape)

                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Track predictions and labels for metrics
                all_labels.extend(target.cpu().numpy().flatten())
                all_predictions.extend(output.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)

        # Calculate metrics from raw outputs and labels
        y_pred_proba = torch.sigmoid(torch.tensor(all_predictions)).numpy()
        y_pred_binary = (y_pred_proba > 0.5).astype(int )

        # Calculate all metrics
        metrics = calculate_metrics(all_labels, y_pred_binary, y_pred_proba)

        return avg_loss, metrics

    def run(self, logger):
        """
        Main training loop.
        """
        num_epochs = self.training_config['training_params'].get('epochs', 10)
        model_save_path = self.training_config['training_params'].get('model_save_path', './model_checkpoints')
        os.makedirs(model_save_path, exist_ok=True)

        pruning_config = self.training_config['compression_params'].get("pruning", {})
        quant_config = self.training_config['compression_params'].get("quantization", {})
        
        # Model Name
        prefixes = []
        if quant_config.get("enabled", False):
            prefixes.append("QAT")
        if pruning_config.get("enabled", False):
            prefixes.append("Pruning")
        model_name = "_".join(prefixes + [self.model.model_name+'.pth'])
        
        # Tracking
        best_val_f1 = -1.0
        best_epoch = -1
        patience = self.training_config['training_params']['early_stopping'].get('patience',5)
        no_improve_epochs = 0

        # Scheduler
        optimizer = self.optimizer
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
        )

        q_start_e = quant_config.get('start_epoch', 3)

        for epoch in range(num_epochs):
            logger.info(f"* Epoch {epoch + 1}/{num_epochs}")
            
            # Apply QAT
            if self.qat and epoch >= q_start_e :
                if pruning_config.get("enabled", False):
                    self.model = remove_pruning(self.model, logger)
                
                self.model.qat_prepared = True
                self.model = prepare_qat_model(
                    self.model,
                    quant_config,
                    logger,
                    q_start_e,
                    epoch
                )
            
            self.model = self.model.to(self.device)
            train_loss = self.train_epoch()

            # Apply pruning
            if pruning_config.get("enabled", False):
                self.model = apply_pruning(self.model, epoch, pruning_config, logger)

            # Validation
            val_loss, metrics = self.validate_epoch()
            f1 = metrics.get('f1_score', None)
            logger.info(f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, F1: {f1:.4f}")

            # LR scheduling
            if schedular is not None and f1 is not None:
                schedular.step(f1)

            #Save best model
            best_val_f1, best_epoch, no_improve_epochs = get_best_model(self.model, f1, epoch, best_val_f1, best_epoch, no_improve_epochs, model_save_path, model_name,self.training_config['compression_params'], logger)

            # Early stopping
            if no_improve_epochs >= patience and self.training_config['training_params']['early_stopping'].get('enabled', False):
                logger.info(f'Early stopping at epoch {epoch+1}. No improvement in F1 for {patience} consecutive epochs.')
                break

        # Cleanup pruning masks
        if pruning_config.get("enabled", False):
            self.model = remove_pruning(self.model, logger)
        
        if self.qat:
            self.model = convert_qat_model(self.model, quant_config, self.val_loader,logger)

        logger.info("Training finished.")

if __name__ == '__main__':
    # Load configuration 
    data_config, model_config, training_config, _, logger, experiment_dir, plots_dir = setup_experiment(log_file_e=False)
    
    # Setup Logger
    logger = setup_logger()

    # Setup Device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Check which model use
    is_snn_model = model_config['snn_model'].get('enabled', False)
    
    # Load DataLoaders
    train_loader, val_loader, test_loader, y_train = get_dataloaders(data_config, training_config, device,logger=logger)

    # Initialize Model
    input_size = train_loader.dataset.features.shape[1] 
    time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps'] if is_snn_model else None
    
    # Load model based on configs
    logger.info('=== Starting ===')
    model = load_model(input_size, time_steps, device, logger, model_config)
    
    # Loss with pos_weight
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=training_config['training_params']['learning_rate'])
    
    # Initialize and Run Trainer
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device, training_config)
    trainer.run(logger=logger)



