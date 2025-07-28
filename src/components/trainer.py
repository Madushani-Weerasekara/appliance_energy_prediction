import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.components.network import LSTMRegression, GRURegression, CNNLSTMRegression


def train_lstm(model, optimizer, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * x_val.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model


def evaluate_model(model, data_loader, y_scaler=None, device='cuda'):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Inverse transform if scalers were used
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
        targets = y_scaler.inverse_transform(targets)
    
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"Evaluation metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


