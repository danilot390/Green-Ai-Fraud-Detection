import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
import os

from src.utils.config_parser import load_config
from src.data.preprocess import FraudDataset 

from models.snn_model import SNNModel 
from models.conventional_model import ConventionalNN
# from src.models.hybrid_model import HybridModel
from src.utils.config_parser import load_config
from src.utils.metrics import calculate_metrics


# Load configuration globally 
training_config = load_config('config/training_config.yaml')
data_config = load_config('config/data_config.yaml')
model_config = load_config('config/model_config.yaml')
is_snn = model_config['snn_model']['enabled']

def get_dataloaders(dataset_name="credit_card_fraud"):
    """
    Loads preprocessed tensors and creates PyTorch DataLoaders.
    Handles both Conventional and Spiking data formats.
    """

    batch_size = training_config['training_params']['batch_size']
    num_workers = training_config['training_params']['num_workers']
    
    processed_dir = os.path.join('data/processed', dataset_name)
    
    # Load tensors
    X_train = torch.load(os.path.join(processed_dir, 'X_train.pt'))
    y_train = torch.load(os.path.join(processed_dir, 'y_train.pt'))
    X_val = torch.load(os.path.join(processed_dir, 'X_val.pt'))
    y_val = torch.load(os.path.join(processed_dir, 'y_val.pt'))
    X_test = torch.load(os.path.join(processed_dir, 'X_test.pt'))
    y_test = torch.load(os.path.join(processed_dir, 'y_test.pt'))

    # Determine sequence_length and time_steps based on model type
    sequence_length = None
    time_steps = None
    if is_snn:
        # For SNN models to reshape the data
        sequence_length = data_config['preprocessing_params']['sequence_length']
        time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps']

    # Create Datasets and DataLoaders
    train_dataset = FraudDataset(X_train, y_train, sequence_length, time_steps)
    val_dataset = FraudDataset(X_val, y_val, sequence_length, time_steps)
    test_dataset = FraudDataset(X_test, y_test, sequence_length, time_steps)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device, training_config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.training_config = training_config

    def train_epoch(self):
        """
        Runs one full epoch of training.
        """
        self.model.train()
        total_loss = 0

        # Reset the SNN's state before each epoch
        if hasattr(self.model, 'reset_membranes'):
            self.model.reset_membranes()

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self):
        """
        Runs one full epoch of validation.
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Reset the SNN's state for each batch
                if hasattr(self.model, 'reset_membranes'):
                    self.model.reset_membranes()

                output = self.model(data)
                loss = self.loss_fn(output, target)
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

    def run(self):
        """
        Main training loop.
        """
        num_epochs = self.training_config['training_params']['epochs']
        model_save_path = self.training_config['training_params']['model_save_path']
        best_val_f1 = -1.0 # Initialize with a value lower than any possible F1 score

        os.makedirs(model_save_path, exist_ok=True)
        
        print(f"Training started for {num_epochs} epochs on {self.device}.")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate_epoch()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val F1 Score: {metrics['f1_score']:.4f}")
            
            from numpy import nan as NAN
            if metrics['auc_roc'] is not NAN:
                print(f"Val AUC-ROC Score: {metrics['auc_roc']:.4f}")
                
            # Save the best model based on F1-score
            if metrics['f1_score'] > best_val_f1:
                best_val_f1 = metrics['f1_score']
                torch.save(self.model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                print(f"New best model saved with F1-score: {best_val_f1:.4f}")

            
        print("Training finished.")

if __name__ == '__main__':

    # Setup Device
    device_str = training_config['training_params']['device']
    device = torch.device(
                'cuda' if torch.cuda.is_available() and device_str == 'cuda'
                else ('mps' if device_str == 'mps' and torch.backends.mps.is_available()
                    else 'cpu')
            )
    print(f"Using device: {device}")
    
    # Load DataLoaders
    # You would have previously run make_dataset.py to generate these
    train_loader, val_loader, _ = get_dataloaders(dataset_name="credit_card_fraud")

    # Initialize Model
    # Handle the different models from config
    input_size = train_loader.dataset.features.shape[1] 
    time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps'] if is_snn else None

    # This part requires a bit of logic based on your model_config
    if model_config['snn_model']['enabled'] and model_config['conventional_nn_model']['enabled']:
        # This is for the Hybrid Model
        # model = HybridModel(input_size=input_size, config=model_config).to(device)
        pass
    elif model_config['snn_model']['enabled']:
        model = SNNModel(input_size=input_size, time_steps=time_steps, config=model_config).to(device)
    else:
        model = ConventionalNN(input_size=input_size, config=model_config).to(device)

    # Initialize Loss Function and Optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=training_config['training_params']['learning_rate'])
    
    # Initialize and Run Trainer
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, device, training_config)
    trainer.run()

