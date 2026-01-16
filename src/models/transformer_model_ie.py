import math
from typing import Mapping

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer base learner following Ileberi & Sun (2024).

    Role:
    - Capture long-range and global dependencies in transaction sequences
    - Act as an independent estimator in a stacking ensemble
    """

    def __init__(
        self,
        input_size: int,
        config: Mapping | None = None,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        output_size: int = 1,
    ):
        super().__init__()

        self.model_name = type(self).__name__

        if config:
            d_model = int(config.get("d_model", d_model))
            num_heads = int(config.get("num_heads", num_heads))
            num_layers = int(config.get("num_layers", num_layers))
            dim_feedforward = int(
                config.get("feedforward_dim", config.get("dim_feedforward", dim_feedforward))
            )
            dropout = float(config.get("dropout_rate", config.get("dropout", dropout)))
            activation = str(config.get("activation", activation)).lower()
            output_size = int(config.get("output_size", output_size))
        else:
            activation = activation.lower()

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        x:
          - (B, T, F)  sequential input
          - (B, F)     tabular input → expanded to sequence length 1
        """

        # Convert (B, F) → (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)  # (B, T, d_model)

        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.encoder(x)

        # Mean pooling over time (global context)
        x = x.mean(dim=1)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        """
        Returns fraud probability P(y=1)
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)
