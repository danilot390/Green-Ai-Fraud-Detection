from xgboost import XGBClassifier
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional
from models.hybrid_model import HybridModel

class HybridStackingModel:
    """
    Stacking architecture:
      - Base learner: HybridModel (SNN + MLP in PyTorch)
      - Meta-learner: XGBoost (operates on fused features or base logits)
    """
    def __init__(self, hybrid_model: HybridModel, xgb_params: dict = None):
        self.hybrid_model = hybrid_model
        self.model_name = type(self).__name__ 
        if xgb_params is None:
            print('Default xgb params used.')
            xgb_params = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            }
        self.meta_model = XGBClassifier(**xgb_params)
        self.device = next(hybrid_model.parameters()).device

    def extract_features_numpy(self, x_tensor: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """
        Runs the HybridModel in eval mode to get fused features and returns them as a NumPy array.
        """
        self.hybrid_model.eval()
        all_features = []

        # Ensure tensor is on the correct device
        x_tensor = x_tensor.to(self.device)
        with torch.no_grad():
            for i in range(0, x_tensor.size(0), batch_size):
                batch = x_tensor[i:i+batch_size]
                feats = self.hybrid_model(batch, return_features=True)  # [B, D]
                all_features.append(feats.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)  # [N, D]

    def fit_meta(self, x_train_tensor: torch.Tensor, y_train: np.ndarray, batch_size: int = 512):
        """
        Trains the XGBoost meta-learner on top of the HybridModel fused features.
        """
        X_meta = self.extract_features_numpy(x_train_tensor, batch_size=batch_size)
        self.meta_model.fit(X_meta, y_train)

    def predict_proba(self, x_tensor: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """
        Predicts fraud probabilities using the stacking setup.
        """
        X_meta = self.extract_features_numpy(x_tensor, batch_size=batch_size)
        proba = self.meta_model.predict_proba(X_meta)[:, 1]  # probability of positive class
        return proba

    def predict_label(self, x_tensor: torch.Tensor, threshold: float = 0.5, batch_size: int = 512) -> np.ndarray:
        """
        Predicts binary labels using a specified probability threshold.
        """
        proba = self.predict_proba(x_tensor, batch_size=batch_size)
        return (proba >= threshold).astype(int)