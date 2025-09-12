import torch
import os
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils.config_parser import load_config
from src.data.preprocess import FraudDataset 

def get_dataloaders(data_config, training_config, device, dataset_name="credit_card_fraud", logger=None):
    """
    Loads preprocessed tensors and creates PyTorch DataLoaders with WeightedRandomSampler if imbalance exists.
    Handles both non-sequential (Conventional NN) and sequential (Spiking NN and Hybrid) data formats.
    """
    model_config = load_config('config/model_config.yaml')
    is_snn_model = model_config['snn_model'].get('enabled', False)
    is_hybrid_model = is_snn_model and model_config['conventional_nn_model'].get('enabled', False)

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