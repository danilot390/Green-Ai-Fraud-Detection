import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
import os
import sys

# Add the parent directory to the system path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_parser import load_config
from src.data.preprocess import FraudDataset 
from src.models.hybrid_model import HybridModel 
from src.utils.metrics import calculate_metrics

def get_dataloaders(dataset_path, sequence_length=None, batch_size=32):
    """
    Loads preprocessed tensors and creates PyTorch DataLoaders.
    """
    # Load tensors
    X_train = torch.load(os.path.join(dataset_path, 'X_train.pt'))
    y_train = torch.load(os.path.join(dataset_path, 'y_train.pt'))
    X_val = torch.load(os.path.join(dataset_path, 'X_val.pt'))
    y_val = torch.load(os.path.join(dataset_path, 'y_val.pt'))
    X_test = torch.load(os.path.join(dataset_path, 'X_test.pt'))
    y_test = torch.load(os.path.join(dataset_path, 'y_test.pt'))

    # Instantiate datasets using your provided FraudDataset class
    train_dataset = FraudDataset(X_train, y_train, sequence_length=sequence_length)
    val_dataset = FraudDataset(X_val, y_val, sequence_length=sequence_length)
    test_dataset = FraudDataset(X_test, y_test, sequence_length=sequence_length)

    # Handle class imbalance with WeightedRandomSampler
    # Get the label for each sequence (the last label in the sequence)
    train_labels = torch.tensor([item[1][-1].item() for item in train_dataset])
    
    class_counts = torch.bincount(train_labels.long())
    if len(class_counts) > 1:
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_labels.long()]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        print("Warning: Only one class found in training data. Using standard DataLoader.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_labels

class Trainer:
    def __init__(self, training_config, data_config, model_config):
        """
        Initializes the trainer with the given configuration.
        """
        self.training_config = training_config
        self.data_config = data_config
        self.model_config = model_config

        snn_time_steps = data_config['preprocessing_params']['snn_input_encoding']['time_steps']
        processed_dir = os.path.join('data/processed/credit_card_fraud')
        
        self.train_loader, self.val_loader, self.test_loader, _ = get_dataloaders(
            dataset_path = processed_dir,
            sequence_length=snn_time_steps,
            batch_size=training_config['training_params']['batch_size']
        )

        # 1. Instantiate the Model
        snn_input_size = self.train_loader.dataset.features.shape[1]
        
        # Use the model configuration from the loaded file
        self.model = HybridModel(snn_input_size, snn_time_steps, self.model_config)
        
        # 2. Define the Loss Function and Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_config['training_params']['learning_rate'])
        

    def train_epoch(self):
        """
        Runs one full epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            target_label = labels[:, -1, :].reshape(outputs.shape)
            loss = self.criterion(outputs, target_label)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        
        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Runs one full epoch of validation.
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                target_label = labels[:, -1, :].reshape(outputs.shape)
                loss = self.criterion(outputs, target_label)
                total_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                all_labels.extend(target_label.cpu().numpy().flatten())
                all_predictions.extend(outputs.cpu().numpy().flatten())

        avg_loss = total_loss / len(self.val_loader)
        
        # Convert raw outputs to probabilities and binary predictions
        y_pred_proba = torch.sigmoid(torch.tensor(all_predictions)).numpy()
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics from the `metrics` file
        metrics = calculate_metrics(all_labels, y_pred_binary, y_pred_proba)

        return avg_loss, metrics

    def run(self):
        """
        Runs the full training loop with validation and model saving.
        """
        num_epochs = self.training_config['training_params']['epochs']
        model_save_path = training_config['training_params']['model_save_path']
        model_name = training_config['training_params']['best_model_hybrid_filename']
        best_val_f1 = -1.0 
        
        os.makedirs(model_save_path, exist_ok=True)
        
        print("Starting training...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate_epoch()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val F1 Score: {metrics['f1_score']:.4f}")
            
            if metrics.get('auc_roc') is not None:
                print(f"Val AUC-ROC Score: {metrics['auc_roc']:.4f}")
            
            # Save the best model based on F1-score
            if metrics['f1_score'] > best_val_f1:
                best_val_f1 = metrics['f1_score']
                torch.save(self.model.state_dict(), os.path.join(model_save_path, model_name))
                print(f"New best model saved with F1-score: {best_val_f1:.4f}")

        print("Training finished!")
        
if __name__ == "__main__":
    # Load configuration files
    training_config = load_config('config/training_config.yaml')
    data_config = load_config('config/data_config.yaml')
    model_config = load_config('config/model_config.yaml')

    # Setup Device
    device_str = training_config['training_params']['device']
    device = torch.device(
                'cuda' if torch.cuda.is_available() and device_str == 'cuda'
                else ('mps' if device_str == 'mps' and torch.backends.mps.is_available()
                    else 'cpu')
            )
    print(f"Using device: {device}")

    # Pass the loaded configurations directly to the Trainer
    trainer = Trainer(training_config, data_config, model_config)
    trainer.run()
