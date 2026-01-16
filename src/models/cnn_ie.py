import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    CNN base learner following Ileberi & Sun (2024).

    Role:
    - Capture spatial/statistical feature interactions
    - Act as an independent estimator in a stacking ensemble
    """

    def __init__(self, input_features: int):
        super().__init__()

        self.model_name = type(self).__name__

        # Convolutional layers 
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

        #  Fully connected layers 
        conv_out_size = input_features // 2 // 2 // 2
        self.fc1 = nn.Linear(128 * conv_out_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        x:
          - (B, F)     tabular input
          - (B, T, F)  sequential input → mean pooled
        """

        # Convert (B, T, F) → (B, F)
        if x.dim() == 3:
            x = x.mean(dim=1)

        # (B, F) → (B, 1, F)
        x = x.unsqueeze(1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)

        return logits.squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x):
        """
        Returns fraud probability P(y=1)
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)
