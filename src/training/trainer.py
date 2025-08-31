import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from torch.optim import Adam
import os

from src.data.preprocess import FraudDataset 
from src.models.snn_model import SNNModel 
from src.models.conventional_model import ConventionalNN
from src.models.hybrid_model import HybridModel
from src.utils.config_parser import load_config
from src.utils.common import get_device
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger

def get_dataloaders(data_config, training_config, device, dataset_name="credit_card_fraud",is_snn_model=False, is_hybrid_model=False, logger=None):
    """
    Loads preprocessed tensors and creates PyTorch DataLoaders with WeightedRandomSampler if imbalance exists.
    Handles both non-sequential (Conventional NN) and sequential (Spiking NN and Hybrid) data formats.
    """
  
    processed_dir = os.path.join(
        data_config['processed_data_paths'].get('credit_card_fraud_path')\
        if dataset_name == "credit_card_fraud" \
        else data_config['processed_data_paths'].get('synthetic_data_path'))
    
    map_location = None
    # Load tensors
    X_train = torch.load(os.path.join(processed_dir, 'X_train.pt'), map_location=map_location)
    y_train = torch.load(os.path.join(processed_dir, 'y_train.pt'), map_location=map_location)
    X_val = torch.load(os.path.join(processed_dir, 'X_val.pt'), map_location=map_location)
    y_val = torch.load(os.path.join(processed_dir, 'y_val.pt'), map_location=map_location)
    X_test = torch.load(os.path.join(processed_dir, 'X_test.pt'), map_location=map_location)
    y_test = torch.load(os.path.join(processed_dir, 'y_test.pt'), map_location=map_location)
    logger.info(f"Data loaded from {processed_dir} ...")

    # Determine batch size, num_workers, sequence_length, and time_steps based on settings
    # For SNN or Hybrid models to reshape the data
    batch_size = training_config['training_params'].get('batch_size', 0)
    num_workers = training_config['training_params'].get('num_workers', 0)
    sequence_length =data_config['preprocessing_params'].get('sequence_length') if is_snn_model else None
    time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps'] if is_snn_model else None

    # Create Datasets and DataLoaders
    train_dataset = FraudDataset(X_train, y_train, sequence_length, time_steps)
    val_dataset = FraudDataset(X_val, y_val, sequence_length, time_steps)
    test_dataset = FraudDataset(X_test, y_test, sequence_length, time_steps)

    # Extract labels for WeightedRandomSampler
    train_labels = torch.tensor([item[1][-1].item() for item in train_dataset])if len(train_dataset[0][1].shape) > 1 else y_train
    
    # WeightedRandomSampler for class imbalance handling
    if is_hybrid_model or is_snn_model:
        class_counts = torch.bincount(train_labels.long())
        if len(class_counts) > 1:
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[train_labels.long()]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
            logger.info("Handle class imbalance with WeightedRandomSampler.")
        else:
            logger.warning("Warning: Only one class found in training data. Using standard DataLoader.")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        logger.info("Using standard DataLoader.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    logger.info("DataLoaders created.")
    return train_loader, val_loader, test_loader, train_labels.to(device)

class Trainer:
    def __init__(self, model,criterion, optimizer, train_loader, val_loader, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.training_config = config


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
            if isinstance(self.model, HybridModel):
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
                if isinstance(self.model, HybridModel):
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
        model_name = self.model.model_name
        best_val_f1 = -1.0 # Initialize with a value lower than any possible F1 score

        os.makedirs(model_save_path, exist_ok=True)
        logger.info(f"{model_name} training for {num_epochs} epochs on {self.device} ...")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate_epoch()
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val F1 Score: {metrics['f1_score']:.4f}")
            
            from numpy import nan as NAN
            if metrics['auc_roc'] is not NAN:
                logger.info(f"Val AUC-ROC Score: {metrics['auc_roc']:.4f}")
                
            # Save the best model based on F1-score
            if metrics['f1_score'] > best_val_f1:
                best_val_f1 = metrics['f1_score']
                torch.save(self.model.state_dict(), os.path.join(model_save_path, f'best_{model_name}.pth'))
                logger.info(f"New best model saved with F1-score: {best_val_f1:.4f}")

        logger.info("Training finished.")

if __name__ == '__main__':
    # Load configuration globally 
    config={
        'training_config' : load_config('config/training_config.yaml'),
        'data_config' : load_config('config/data_config.yaml'),
        'model_config' : load_config('config/model_config.yaml'),
    }
    
    # Setup Logger
    logger = setup_logger()

    # Setup Device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Check which model use
    is_snn_model = config['model_config']['snn_model']['enabled']
    is_convetional_model = config['model_config']['conventional_nn_model']['enabled']
    is_hybrid_model = is_snn_model and is_convetional_model
    
    
    # Load DataLoaders
    # You would have previously run make_dataset.py to generate these
    train_loader, val_loader, test_loader, y_train = get_dataloaders(
        dataset_name="credit_card_fraud",
        data_config=config['data_config'],
        training_config=config['training_config'],
        device=device,
        is_snn_model=is_snn_model,
        is_hybrid_model=is_hybrid_model,
        logger=logger,
        )

    # Initialize Model
    # Handle the different models from config
    input_size = train_loader.dataset.features.shape[1] 
    time_steps = config['data_config']['preprocessing_params']['snn_input_encoding']['time_steps'] if is_snn_model else None

    # This part requires a bit of logic based on your model_config
    logger.info('=== Neural Network ===')
    if is_hybrid_model:
        model = HybridModel(snn_input_size=input_size, snn_time_steps=time_steps, config=config['model_config']).to(device)
    elif is_snn_model:
        model = SNNModel(input_size=input_size, time_steps=time_steps, config=config['model_config']).to(device)
    elif is_convetional_model:
        model = ConventionalNN(input_size=input_size, config=config['model_config']).to(device)
    else:
        logger.error("Invalid model configuration. At least one model must be enabled.")
        raise ValueError("INvalid model configuration. At least one model must be enabled.")

    # Loss with pos_weight
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], device=device, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=config['training_config']['training_params']['learning_rate'])
    
    # Initialize and Run Trainer
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, device, config['training_config'])
    trainer.run(logger=logger)

