import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

class XBoost_CNN_BiLSTM(nn.Module):
    """ 
    Hybrid deep learning model combining CNN, BiLSTM, and MLP architectures, 
    with optional integration of XGBoost-derived embeddings for tabular data.
    """
    def __init__(self, input_size, cnn_config, lstm_hidden=64, mlp_config=None):
        super().__init__()
        self.input_size = input_size
        self.model_name = type(self).__name__ 

        # ----- CNN Branch -----
        cnn_layers = []
        in_channels = 1
        self.seq_len = input_size
        for layer_cfg in cnn_config['cnn_layers']:
            cnn_layers.append(nn.Conv1d(in_channels, layer_cfg['out_channels'], kernel_size=layer_cfg['kernel_size'], padding=layer_cfg.get('padding',0)))
            cnn_layers.append(nn.ReLU())
            if 'pool_size' in layer_cfg:
                cnn_layers.append(nn.MaxPool1d(layer_cfg['pool_size']))
            in_channels = layer_cfg['out_channels']
        self.cnn_stack = nn.Sequential(*cnn_layers)
        with torch.no_grad():
            x = torch.zeros(1,1,self.seq_len)
            cnn_out_size = self.cnn_stack(x).view(1,-1).shape[1]

        # ----- BiLSTM Branch -----
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True, bidirectional=True)

        # ----- MLP classifier -----
        fusion_input = cnn_out_size + lstm_hidden*2 + input_size  # include XGBoost outputs
        mlp_layers = []
        in_features = fusion_input
        for layer_params in mlp_config['mlp_layers']:
            mlp_layers.append(nn.Linear(in_features, layer_params['units']))
            if layer_params.get('batchnorm', False):
                mlp_layers.append(nn.BatchNorm1d(layer_params['units']))
            if layer_params['activation'] == 'ReLU':
                mlp_layers.append(nn.ReLU())
            elif layer_params['activation'] == 'Tanh':
                mlp_layers.append(nn.Tanh())
            if 'dropout_rate' in layer_params:
                mlp_layers.append(nn.Dropout(layer_params['dropout_rate']))
            in_features = layer_params['units']
        mlp_layers.append(nn.Linear(in_features, 1))
        self.mlp_stack = nn.Sequential(*mlp_layers)

    def forward(self, x, xgb_emb=None):
        # CNN
        x_cnn = x.unsqueeze(1)
        x_cnn = self.cnn_stack(x_cnn)
        x_cnn = x_cnn.view(x.size(0), -1)

        # BiLSTM
        x_lstm, _ = self.lstm(x.unsqueeze(1))
        x_lstm = x_lstm[:, -1, :]  # last output

        # XGBoost embedding (precomputed)
        if xgb_emb is None:
            xgb_emb = x  # fallback if not using XGBoost separately

        # Fusion
        fusion = torch.cat([x_cnn, x_lstm, xgb_emb], dim=1)
        logits = self.mlp_stack(fusion)
        return logits
