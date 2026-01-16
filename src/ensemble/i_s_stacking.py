import numpy as np
import torch
from xgboost import XGBClassifier


class IleberiSunStackingModel:
    """
    Prediction-level stacking ensemble following:
    Ileberi & Sun (2024) - IEEE Access

    Base learners:
      - CNN
      - LSTM
      - Transformer

    Meta learner:
      - XGBoost
    """

    def __init__(
        self,
        cnn_model,
        lstm_model,
        transformer_model,
        device,
        xgb_params=None,
    ):
        self.model_name = type(self).__name__

        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model

        self.device = device

        self.meta_model = XGBClassifier(
            **(xgb_params or {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "use_label_encoder": False,
            })
        )

    def to(self, device):
        self.device = device
        self.cnn_model.to(device)
        self.lstm_model.to(device)
        self.transformer_model.to(device)
        return self

    # Extract meta-features
    @torch.no_grad()
    def _meta_features_from_loader(self, data_loader):
        self.cnn_model.eval()
        self.lstm_model.eval()
        self.transformer_model.eval()

        meta_features = []
        labels = []

        for x, y in data_loader:
            x = x.to(self.device)

            p_cnn = self.cnn_model.predict_proba(x).view(-1)
            p_lstm = self.lstm_model.predict_proba(x).view(-1)
            p_trans = self.transformer_model.predict_proba(x).view(-1)

            stacked = torch.stack([p_cnn, p_lstm, p_trans], dim=1)
            meta_features.append(stacked.cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        X_meta = np.vstack(meta_features)
        y_meta = np.concatenate(labels)

        return X_meta, y_meta

    def fit_meta_from_loader(self, val_loader):
        """
        Convenience wrapper that expects `val_loader` to iterate over a held-out
        split whose samples were not used to fit the base learners.
        """
        X_meta, y_val = self._meta_features_from_loader(val_loader)
        self.meta_model.fit(X_meta, y_val)

    # Training
    def fit_meta(self, val_loader):
        """
        Train XGBoost meta-learner on validation/OOF predictions.
        Caller must ensure the provided loader never exposes training
        samples used to fit the base learners, mirroring the procedure
        described by Ileberi & Sun (2024).
        """
        X_meta, y_meta = self._meta_features_from_loader(val_loader)
        self.meta_model.fit(X_meta, y_meta)

        return self

    # Inference
    @torch.no_grad()
    def predict_proba(self, x):
        """
        x: (B, T, F) or (B, F)
        Returns fraud probability
        """
        self.cnn_model.eval()
        self.lstm_model.eval()
        self.transformer_model.eval()

        x = x.to(self.device)

        p_cnn = self.cnn_model.predict_proba(x).view(-1)
        p_lstm = self.lstm_model.predict_proba(x).view(-1)
        p_trans = self.transformer_model.predict_proba(x).view(-1)

        X_meta = torch.stack([p_cnn, p_lstm, p_trans], dim=1)
        X_meta = X_meta.cpu().numpy()

        return self.meta_model.predict_proba(X_meta)[:, 1]

    def predict(self, x, threshold=0.5):
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)

    def eval(self):
        return self
    
    def train(self):
        return self
