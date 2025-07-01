"""
Main workflow runner for the Appliance Energy Prediction Pipeline
Supports both classic ML and Deep Learning model pipelines.

Steps:
1. Preprocess & feature engineer
2. Train model (classic ML or deep learning)
3. Evaluate & print results
"""

import numpy as np
import optuna
from src.components.data_transformation import DataTransformation
from src.components.ml_model_trainer import ModelTrainer
from src.logger import setup_logger
from src.exception import CustomException
from src.components.tuner import objective

# Deep Learning/Skorch imports
import torch
import torch.optim as optim
from src.utils import save_object, load_object, save_model, load_model,   create_sequences, evaluate_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle
from skorch.callbacks import EarlyStopping


logger = setup_logger()

def main():
    print(""" ==================================================================
                            starting
          =======================================================================""")
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

    except:
        pass
        
    
    # DL Models 
    if hasattr(objective, "best_score"):
        del objective.best_score
    if hasattr(objective, "best_model_path"):
        del objective.best_model_path

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(f"  Value (RMSE): {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"Best model saved at: {objective.best_model_path}")



if __name__ == "__main__":
    main()
