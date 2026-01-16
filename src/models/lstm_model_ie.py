import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM base learner following Ileberi & Sun (2024).

    Role:
    - Capture temporal / sequential dependencies in transactions
    - Act as an independent estimator in a stacking ensemble
    """

    def __init__(self, input_size: int, hidden_sizes=(50, 100), dropout=0.5):
        super().__init__()

        self.model_name = type(self).__name__

        #  LSTM layers 
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[-1],
            num_layers=len(hidden_sizes),
            dropout=dropout if len(hidden_sizes) > 1 else 0.0,
            batch_first=True
        )

        # Connected layer 
        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        """
        x:
          - (B, T, F)  sequential input
          - (B, F)     tabular input → expanded to sequence length 1
        """

        # Convert (B, F) → (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # LSTM forward
        out, _ = self.lstm(x)

        # Use last timestep 
        out = out[:, -1, :]

        logits = self.fc(out)
        return logits.squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x):
        """
        Returns fraud probability P(y=1)
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)