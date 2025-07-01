import torch
from torch.utils.data import Dataset
import numpy as np



class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    


def create_sequences(df, seq_len=24, target_col='target'):
    data = df.values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :-1])    # all feature columns, last 'seq_len' timesteps
        y.append(data[i+seq_len, -1])       # target value AFTER the window
    return np.array(X), np.array(y).reshape(-1, 1)
    

#dataset.py

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_loaders_from_dataframe(
    df, target_col='Appliances_capped', seq_len=10, batch_size=64, val_size=0.2, shuffle=True
    ):
    df = df.drop(columns=['date'])
    # Move target to last column for easy slicing
    cols = [col for col in df.columns if col != target_col] + [target_col]
    df = df[cols]

    print("Columns:", df.columns)
    print("Target col used:", target_col) 

    # Feature scaling (fit only on train)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    # Temporarily create sequences without scaling for splitting
    X, y = create_sequences(df, seq_len=seq_len, target_col=target_col)

    # Flatten for splitting (so split doesn't break sequences)
    idxs = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(idxs, test_size=val_size, shuffle=shuffle, random_state=42)
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    # Fit scalers on train
    X_train = X[train_idx].reshape(-1, X.shape[-1])
    y_train = y[train_idx]
    scaler_x.fit(X_train)
    scaler_y.fit(y_train)

    # Scale all
    X = X.reshape(-1, X.shape[-1])
    X = scaler_x.transform(X)
    X = X.reshape(-1, seq_len, X.shape[-1])
    y = scaler_y.transform(y)

    # Now split again
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset   = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, scaler_x, scaler_y
