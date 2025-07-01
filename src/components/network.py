import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skorch import NeuralNetRegressor
import torch.optim as optim
from sklearn.model_selection import GridSearchCV

 

# === LSTM ARCHITECTURE ===
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=64, output_dim=1, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # get the last time step
        out = self.fc(out)
        return out



# === GRU ARCHITECTURE ===
class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=64, output_dim=1, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# === CNN-LSTM HYBRID ARCHITECTURE ===
class CNNLSTMNetwork(nn.Module):
    def __init__(self, input_dim, seq_len, num_filters=32, kernel_size=3, lstm_hidden_size=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.lstm = nn.LSTM(num_filters, lstm_hidden_size, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_size, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # to (batch, input_dim, seq_len) for Conv1d
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # back to (batch, seq_len', num_filters)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
