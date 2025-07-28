import optuna
import torch
import joblib
import pandas as pd

from src.components.dataset import get_loaders_from_dataframe
from src.components.network import LSTMRegression, GRURegression, CNNLSTMRegression
from src.components.trainer import train_lstm, evaluate_model


def objective(trial, model='lstm'):
    df = pd.read_csv("data/processed/preprocessed_data.csv")
    # 1. Hyperparameter suggestions
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Updated for Optuna >=v3
    seq_len = trial.suggest_int('seq_len', 12, 48)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    if model == "cnn-lstm":
        # CNN-LSTM specific hyperparameters
        cnn_out_channels = trial.suggest_int('cnn_out_channels', 16, 64)
        kernel_size = trial.suggest_int('kernel_size', 2, 5)
        lstm_hidden_dim = trial.suggest_int('lstm_hidden_dim', 32, 128)
        lstm_num_layers = trial.suggest_int('lstm_num_layers', 1, 3)

    # 2. Prepare loaders (always use the correct target_col!)
    train_loader, val_loader, x_scaler, y_scaler = get_loaders_from_dataframe(
        df,
        target_col="Appliances_capped",
        seq_len=seq_len,
        batch_size=batch_size,
        val_size=0.2
    )

    input_dim = train_loader.dataset.X.shape[-1]
    output_dim = train_loader.dataset.y.shape[-1]

    # 3. Build and train model
    if model == 'lstm':
        model = LSTMRegression(input_dim=input_dim, hidden_dim=hidden_dim,
                           num_layers=num_layers, output_dim=output_dim, dropout=dropout)
    elif model == "gru":
        model = GRURegression(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout=dropout)

    elif model == "cnn-lstm":
        model = CNNLSTMRegression(input_dim=input_dim, cnn_out_channels=cnn_out_channels, kernel_size=kernel_size, 
                                  lstm_hidden_dim=lstm_hidden_dim, lstm_num_layers=lstm_num_layers, output_dim=output_dim,
                                    dropout=dropout
    )

    model = train_lstm(model, train_loader, val_loader, num_epochs=50, lr=lr, device='cuda')

    # 4. Evaluate
    metrics = evaluate_model(model, val_loader, y_scaler, device='cuda')
    score = metrics['rmse']  # or 'mae' or 'mse' if preferred

    # 5. Save best model/scalers (only for the best trial)
    if not hasattr(objective, "best_score") or score < objective.best_score:
        print(f"\nNew best model at trial {trial.number}: RMSE={score:.4f}")
        save_path = f"artifacts\\best_{model.name}_model_trial_{trial.number}_rmse{score:.4f}.pt"

        torch.save(model.state_dict(), save_path)
        joblib.dump(x_scaler, f"x_scaler_trial.pkl")
        joblib.dump(y_scaler, f"y_scaler_trial.pkl")
        objective.best_score = score
        objective.best_model_path = save_path

    return score
