import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMRegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.name = "LSTM Regression"
        self.to(device=device)


    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # out: [batch, seq_len, hidden_dim]
        out = out[:, -1, :]  # Take only last time step's output
        out = self.fc(out)
        return out



class GRURegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRURegression, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.name = "GRU Regression"
        self.to(device=device)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)  # out: [batch, seq_len, hidden_dim]
        out = out[:, -1, :]       # Take only the last time step's output
        out = self.fc(out)
        return out



class CNNLSTMRegression(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, kernel_size, lstm_hidden_dim, 
                 lstm_num_layers, output_dim, dropout=0.2):
        super(CNNLSTMRegression, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, 
                             out_channels=cnn_out_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=cnn_out_channels, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_num_layers,
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.name = "CNN-LSTM Regression"
        self.to(device=device)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # -> [batch, input_dim, seq_len]
        x = self.cnn(x)          # -> [batch, cnn_out_channels, seq_len]
        x = self.relu(x)
        x = x.permute(0, 2, 1)   # -> [batch, seq_len, cnn_out_channels]
        # LSTM expects [batch, seq_len, cnn_out_channels]
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]      # Last time step
        out = self.fc(out)
        return out
    


 
 


