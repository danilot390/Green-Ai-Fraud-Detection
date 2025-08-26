import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import DataLoader, Dataset
import torch
import os

from src.utils.config_parser import load_config

# Load configuration globally for the module
data_config = load_config('config/data_config.yaml')
model_config = load_config('config/model_config.yaml')
is_snn = model_config['snn_model']['enabled']

# Dataset paths
RAW_DATA_PATH = data_config['raw_data_paths']
PROCESSED_DATA_PATH = data_config['processed_data_paths']
SPLIT_RATIOS = data_config['data_ratios']
PREPROCESSING_PARAMS = data_config['preprocessing_params']

def load_raw_data(dataset_name):
    """
    Load raw data from the specified file path.
    """
    if dataset_name not in RAW_DATA_PATH:
        raise ValueError(f"File path '{dataset_name}' is not defined in the raw data paths.")
    
    file_path = RAW_DATA_PATH[dataset_name]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file '{file_path}' does not exist.")
    
    print(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Raw data loaded successfully with shape: {df.shape}")
    return df   

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in the DataFrame.
    """
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        if fill_value is None:
            raise ValueError("Fill value must be provided when using 'fill' strategy.")
        df = df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy. Use 'drop' or 'fill'.")
    
    print(f"Missing values handled. New shape: {df.shape}")
    return df

def feature_engineering(df, dataset_name):
    """
    Apply feature engineering functions to the DataFrame.
    """
    print(f"Performing feature engineering for {dataset_name} dataset...")
    if dataset_name == 'credit_card_fraud':
        # The 'Time' feature is seconds elapsed.
        # For simplicity in this initial step, just drop it if configured.
        if "Time" in df.columns and "Time" in PREPROCESSING_PARAMS.get('drop_columns', []):
            df = df.drop(columns=["Time"])
            print("Dropped 'Time'.")
        # 'Amount' feature might be highly skewed. Consider log transformation.
        if PREPROCESSING_PARAMS.get('log_transfdrop_columnsorm_amount', False):
             df['Amount'] = np.log1p(df['Amount']) # log1p handles zero amounts
             print("Log transformed 'Amount' feature.")

    elif dataset_name == 'synthetic_data':
        # Feature ideas for time-series / sequential data:
        # 1. Transaction frequency for accounts (requires grouping by account)
        # 2. Velocity of spending (average amount over last N steps)
        # 3. Balance changes (already somewhat explicit, but could be ratios)
        # 4. Flags for specific suspicious patterns (e.g., zero balance before large cash-out)

        # Average amount features
        df['orig_avg_amount'] = df.groupby('nameOrig')['amount'].transform('mean')
        df['dest_avg_amount'] = df.groupby('nameDest')['amount'].transform('mean')
        df['amount_vs_avg_sender'] = df['amount'] / (df['orig_avg_amount'] + 1e-9)

        # Simple balance change features
        df['amount_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['amount_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

        # 'nameOrig' and 'nameDest' are dropped to prevent overfitting
        # on specific account IDs and to keep feature space manageable.
        df = df.drop(columns=['nameOrig', 'nameDest'], errors='ignore')
        print("Dropped 'nameOrig' and 'nameDest' features.")

        # Handling 'type' feature (categorical)
        if 'type' in df.columns:
            # One-hot encoding for transaction 'type'
            df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=True)
            print("One-hot encoded 'type' feature.")

        # Drop original balance columns if new diff features are preferred (optional)
        if PREPROCESSING_PARAMS.get('drop_original_balance_cols', False):
            df = df.drop(columns=['oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
            print("Dropped original balance columns.")

    else:
        print(f"No specific feature engineering defined for {dataset_name}.")

    print("Feature engineering complete.")
    return df

def scale_features(X_train, X_val, X_test, numerical_features):
    """    
    Scale numerical features in the training, validation, and test sets.
    """
    scaling_method = PREPROCESSING_PARAMS.get('scaling_method', 'StandardScaler')
    print(f"Scaling numerical features using: {scaling_method}...")

    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_method is None:
        print("No scaling applied.")
        return X_train, X_val, X_test, None
    else:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")

    # Fit scaler only on training data to prevent data leakage
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val_scaled[numerical_features] = scaler.transform(X_val[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    print("Feature scaling complete.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def handle_imbalance(X_train, y_train):
    """
    Applies resampling techniques to handle class imbalance in training data.
    """
    if not PREPROCESSING_PARAMS.get('handle_imbalanced_data', False):
        print("Imbalance handling is disabled.")
        return X_train, y_train

    imbalance_method = PREPROCESSING_PARAMS.get('imbalance_method', 'SMOTE')
    print(f"Handling imbalance using: {imbalance_method}...")

    if imbalance_method == "SMOTE":
        sampling_strategy = PREPROCESSING_PARAMS.get('smote_sampling_strategy')
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=SPLIT_RATIOS['random_state'])

    elif imbalance_method == "ADASYN":
        sampler = ADASYN(random_state=SPLIT_RATIOS['random_state'])
    elif imbalance_method == "RandomUnderSampler":
        # Be cautious with undersampling as it discards data
        sampler = RandomUnderSampler(random_state=SPLIT_RATIOS['random_state'])
    elif imbalance_method == "None":
        print("No resampling applied, but imbalance handling is enabled.")
        return X_train, y_train # Return original if method is None
    else:
        raise ValueError(f"Unsupported imbalance method: {imbalance_method}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"Original training data shape: {X_train.shape}, {y_train.value_counts()}")
    print(f"Resampled training data shape: {X_resampled.shape}, {y_resampled.value_counts()}")
    print("Imbalance handling complete.")
    return X_resampled, y_resampled

def split_data(df, target_column, dataset_name):
    """
    Splits the DataFrame into training, validation, and test sets.
    Uses stratification for general data and a temporal split for the synthetic dataset.
    """
    print("Splitting data into train, validation, and test sets...")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    train_ratio = SPLIT_RATIOS['train']
    val_ratio = SPLIT_RATIOS['validation']
    test_ratio = SPLIT_RATIOS['test']
    random_state = SPLIT_RATIOS['random_state']

    if dataset_name == 'synthetic_data':
        print("Performing a temporal split based on the 'step' column.")
      
        # Find split points based on ratios
        max_step = df['step'].max()
        train_end_step = int(max_step * SPLIT_RATIOS['train'])
        val_end_step = int(max_step * (SPLIT_RATIOS['train'] + SPLIT_RATIOS['validation']))


        # Split the data chronologically
        train_df = df[df['step'] <= train_end_step]
        val_df = df[(df['step'] > train_end_step) & (df['step'] <= val_end_step)]
        test_df = df[df['step'] > val_end_step]
        
        # Remove the 'step' column from features
        X_train, y_train = train_df.drop(columns=[target_column, 'step']), train_df[target_column]
        X_val, y_val = val_df.drop(columns=[target_column, 'step']), val_df[target_column]
        X_test, y_test = test_df.drop(columns=[target_column, 'step']), test_df[target_column]

    else:
        stratify_by = SPLIT_RATIOS['stratify']
        print("Performing a stratified random split.")
        # First split: Training + Validation vs. Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=y if stratify_by else None
        )

        relative_val_ratio = val_ratio / (train_ratio + val_ratio)

        # Second split: Training vs. Validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=relative_val_ratio, random_state=random_state, stratify=y_train_val if stratify_by else None
        )

    print(f"Train set shape: {X_train.shape}, Fraud: {y_train.sum()} ({y_train.mean():.4f}%)")
    print(f"Validation set shape: {X_val.shape}, Fraud: {y_val.sum()} ({y_val.mean():.4f}%)")
    print(f"Test set shape: {X_test.shape}, Fraud: {y_test.sum()} ({y_test.mean():.4f}%)")
    print("Data splitting complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test

# PyTorch Dataset class for handling data loading and batching
class FraudDataset(Dataset):   
    """
    Custom Dataset class for fraud detection data.
    """
    def __init__(self, features, labels, sequence_length=None, time_steps=None):
        """
        Initialize the dataset 
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.time_steps = time_steps # Used if a more complex SNN encoding is done here

        if self.sequence_length is not None:
            # This is a simplified sequentialization.
            # Here, it creates overlapping sequences.
            num_samples = len(self.features)
            assert num_samples >= self.sequence_length, "Dataset too small for given sequence length."
            self.indices = [(i, i + self.sequence_length) for i in range(num_samples - self.sequence_length + 1)]
            print(f"Dataset initialized with {len(self.indices)} sequences of length {self.sequence_length}.")
        else:
            self.indices = list(range(len(self.features)))
            print(f"Dataset initialized with {len(self.indices)} samples.")
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.indices)
    
    def __getitem__(self, idx):
        """            
        Returns a sample from the dataset.
        """
        if self.sequence_length is not None:
            start, end = self.indices[idx]
            features = self.features[start:end]
            labels = self.labels[start:end]
            return features, labels
        else:
            return self.features[idx], self.labels[idx]

def get_preprocessed_data(dataset_name="credit_card_fraud", target_column="Class", strategy='X/y tensors'):
    """
    Run the entire data preprocessing pipeline.
    """
    print(f"\n--- Starting preprocessing for {dataset_name} ---")

    # 1. Load Raw Data
    df = load_raw_data(dataset_name)

    # 2. Handle Missing Values
    df = handle_missing_values(df)

    # 3. Feature Engineering
    df = feature_engineering(df, dataset_name)

    # 4. Split Data (before scaling/resampling to prevent data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column=target_column, dataset_name=dataset_name)

    engineered_numerical_features = X_train.columns.tolist()
    
    # 5. Scale Features (fit on X_train only)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, engineered_numerical_features
    )

    # 6. Handle Imbalance (only on X_train_scaled, y_train)
    X_train_resampled, y_train_resampled = handle_imbalance(X_train_scaled, y_train)
   
    # 7. Convert to Pytorch Tensors
    X_train_tensor = torch.tensor(X_train_resampled.values , dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_resampled.values , dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled.values , dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values , dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled.values , dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values , dtype=torch.float32).unsqueeze(1)

    # Return feature names for model input layer and XAI later
    feature_names = X_train_resampled.columns.tolist()

    print(f"--- Preprocessing for {dataset_name} complete ---")
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, scaler, feature_names

if __name__ == "__main__":
    print("Running preprocess.py as main for testing...")

    # Test Credit Card Fraud Dataset
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = \
            get_preprocessed_data(dataset_name="credit_card_fraud", target_column="Class")
        

    print("\nCredit Card Fraud Tensors created:")
    print(f"First train batch features shape: {next(iter(X_val)).shape}")
    print(f"First train batch labels shape: {next(iter(y_val)).shape}")
    print(f"Number of features: {len(feature_names)}")

    # Test Synthetic Financial Dataset
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names = \
       get_preprocessed_data(dataset_name="synthetic_data", target_column="isFraud") 

    # print("\nSynthetic Financial DataLoaders created:")
    # print(f"First train batch features shape: {next(iter(synthetic_train_loader))[0].shape}")
    # print(f"First train batch labels shape: {next(iter(synthetic_train_loader))[1].shape}")
    # print(f"Number of features: {len(synthetic_feature_names)}")