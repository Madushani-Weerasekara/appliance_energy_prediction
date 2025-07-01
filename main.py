"""
Main workflow runner for the Appliance Energy Prediction Pipeline
Supports both classic ML and Deep Learning model pipelines.

Steps:
1. Preprocess & feature engineer
2. Train model (classic ML or deep learning)
3. Evaluate & print results
"""

import numpy as np
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import setup_logger
from src.exception import CustomException

# Deep Learning/Skorch imports
import torch
from skorch import NeuralNetRegressor
import torch.optim as optim
from src.components.network import LSTMNetwork, GRUNetwork, CNNLSTMNetwork
from src.utils import save_object, load_object, save_model, load_model,   create_sequences, evaluate_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle
from skorch.callbacks import EarlyStopping


logger = setup_logger()

def main():
    try:
        logger.info("=== Appliance Energy Prediction Pipeline Started ===")

        # === 1. Data Transformation (Preprocessing) ===
        logger.info("Step 1: Running data transformation...")
        transformer = DataTransformation()
        transformer.run(raw_path="data/raw/energy_data_set.csv")
        logger.info("Data preprocessing completed successfully.")

        # === 2. Load Processed Data ===
        train_arr = np.load("data/processed/train_arr.npy")
        test_arr = np.load("data/processed/test_arr.npy")
        X_train = train_arr[:, :-1]
        y_train = train_arr[:, -1].reshape(-1, 1)
        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1].reshape(-1, 1)

        # === 3. Choose Classic ML or Deep Learning Model ===
        PIPELINE_TYPE = "DL"   # "ML" for classic scikit-learn pipeline, "DL" for deep learning

        # === Target Scaling for DL only ===
        if PIPELINE_TYPE == "DL":
            y_scaler = MinMaxScaler()
            y_train_scaled = y_scaler.fit_transform(y_train)
            y_test_scaled = y_scaler.transform(y_test)
            # Save scaler for inference
            with open("artifacts/target_scaler.pkl", "wb") as f:
                pickle.dump(y_scaler, f)
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test

        if PIPELINE_TYPE == "ML":
            logger.info("Using classic ModelTrainer pipeline (scikit-learn models).")
            model_trainer = ModelTrainer()
            final_r2, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
            save_model(best_model, "artifacts/best_ml_model.pkl")
            logger.info("Best ML model saved to artifacts/best_ml_model.pkl")
            logger.info(f"Pipeline finished successfully! Final test R² score: {final_r2:.4f}")
            print(f"Final test R² score: {final_r2:.4f}")

        elif PIPELINE_TYPE == "DL":
            # ---- Deep Learning pipeline using skorch ----
            MODEL_TYPE = 'LSTM'  # Options: 'LSTM', 'GRU', 'CNN-LSTM'
            SEQ_LEN = 10        # For sequence models

            if MODEL_TYPE == 'LSTM':
                logger.info("Using LSTM for time-series regression.")
                X_train_seq, y_train_seq = create_sequences(X_train, y_train_scaled, SEQ_LEN)
                X_test_seq, y_test_seq = create_sequences(X_test, y_test_scaled, SEQ_LEN)
                net = NeuralNetRegressor(
                    LSTMNetwork,
                    module__input_dim=X_train.shape[1],
                    module__output_dim=1,
                    module__hidden_size=64,
                    optimizer=optim.Adam,
                    max_epochs=20,
                    lr=0.01,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    callbacks=[EarlyStopping(patience=5)]
                )
                X_train_fit = X_train_seq.astype(np.float32)
                y_train_fit = y_train_seq.astype(np.float32)
                X_test_fit = X_test_seq.astype(np.float32)
                y_test_fit = y_test_seq.astype(np.float32)

            elif MODEL_TYPE == 'GRU':
                logger.info("Using GRU for time-series regression.")
                X_train_seq, y_train_seq = create_sequences(X_train, y_train_scaled, SEQ_LEN)
                X_test_seq, y_test_seq = create_sequences(X_test, y_test_scaled, SEQ_LEN)
                net = NeuralNetRegressor(
                    GRUNetwork,
                    module__input_dim=X_train.shape[1],
                    module__output_dim=1,
                    module__hidden_size=64,
                    optimizer=optim.Adam,
                    max_epochs=20,
                    lr=0.01,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    callbacks=[EarlyStopping(patience=5)]
                )
                X_train_fit = X_train_seq.astype(np.float32)
                y_train_fit = y_train_seq.astype(np.float32)
                X_test_fit = X_test_seq.astype(np.float32)
                y_test_fit = y_test_seq.astype(np.float32)

            elif MODEL_TYPE == 'CNN-LSTM':
                logger.info("Using CNN-LSTM hybrid for time-series regression.")
                X_train_seq, y_train_seq = create_sequences(X_train, y_train_scaled, SEQ_LEN)
                X_test_seq, y_test_seq = create_sequences(X_test, y_test_scaled, SEQ_LEN)
                net = NeuralNetRegressor(
                    CNNLSTMNetwork,
                    module__input_dim=X_train.shape[1],
                    module__seq_len=SEQ_LEN,
                    module__output_dim=1,
                    module__num_filters=32,
                    module__kernel_size=3,
                    module__lstm_hidden_size=64,
                    optimizer=optim.Adam,
                    max_epochs=20,
                    lr=0.01,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    callbacks=[EarlyStopping(patience=5)]
                )
                X_train_fit = X_train_seq.astype(np.float32)
                y_train_fit = y_train_seq.astype(np.float32)
                X_test_fit = X_test_seq.astype(np.float32)
                y_test_fit = y_test_seq.astype(np.float32)

            else:
                raise ValueError("Invalid MODEL_TYPE selected.")

            # === Grid Search ===
            logger.info("Starting hyperparameter tuning with GridSearchCV...")
            params = {
                'module__hidden_size': [64, 128],  # Works for MLP/LSTM/GRU
                'module__dropout': [0.0, 0.2, 0.4],
                'lr': [0.01, 0.001],
                'max_epochs': [20, 30],
                'batch_size': [32, 64, 128],
                'module__num_layers': [1, 2]

            }
            gs = GridSearchCV(net, params, refit=True, cv=3, scoring='r2', verbose=2)
            gs.fit(X_train_fit, y_train_fit)

            print(f"Best params: {gs.best_params_}")
            print(f"Best R² score (CV val): {gs.best_score_:.4f}")

            # === Test Set Evaluation (Inverse transform) ===
            y_pred = gs.predict(X_test_fit)
            with open("artifacts/target_scaler.pkl", "rb") as f:
                y_scaler = pickle.load(f)
            y_pred_real = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_real = y_scaler.inverse_transform(y_test_fit.reshape(-1, 1))

            test_r2 = r2_score(y_test_real, y_pred_real)
            test_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
            mae = mean_absolute_error(y_test_real, y_pred_real)
            mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
            print(f"Final test R² score: {test_r2:.4f}")
            print(f"Final test RMSE: {test_rmse:.4f}")
            print(f"Final test MAE: {mae:.4f}")
            print(f"Final test MAPE: {mape:.2f}%")

            print("="*40)
            print(f"Best Grid Search Params: {gs.best_params_}")
            print(f"CV R² Score: {gs.best_score_:.4f}")
            print(f"Test R² Score: {test_r2:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print("="*40)

            # === Save the best model ===
            best_model = gs.best_estimator_
            torch.save(best_model, "artifacts/dl_model.pt")
            logger.info("Best deep learning model saved to artifacts/best_deep_model.pkl")

            logger.info("=== Appliance Energy Prediction Pipeline Completed ===")

        else:
            raise ValueError("PIPELINE_TYPE must be 'ML' or 'DL'")

    except CustomException as ce:
        logger.error(f"CustomException in pipeline: {ce}")
        print(f"CustomException: {ce}")
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        print(f"Unexpected error: {e}")



if __name__ == "__main__":
    main()
