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
from functools import partial
from src.logger import logger

def main():
    print(""" ==================================================================
                            starting
          =======================================================================""")
    try:
        logger.info("=== Appliance Energy Prediction Pipeline Started ===")

        # === 1. Data Transformation (Preprocessing) ===
        logger.info("Step 1: Running data transformation...")
        transformer = DataTransformation()
        transformer.run_dl(raw_path="data/raw/energy_data_set.csv")
        logger.info("Data preprocessing completed successfully.")

            
        
        # DL Models 
        if hasattr(objective, "best_score"):
            del objective.best_score
        if hasattr(objective, "best_model_path"):
            del objective.best_model_path

        study = optuna.create_study(direction="minimize")
        #study.optimize(objective, n_trials=5)
        study.optimize(partial(objective, model='lstm'), n_trials=10)


        print("Best trial:")
        print(f"  Value (RMSE): {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        print(f"Best model saved at: {objective.best_model_path}")

    except Exception as e:
        logger.info(f"Error occurred during pipeline execution. {e}")


if __name__ == "__main__":
    main()
