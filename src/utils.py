# src/utils.py

import os
import sys
import pickle   # For saving/loading Python objects (models, transformers)
import joblib   # For saving/loading scikit-learn/skorch models (optional)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import setup_logger

# Initialize logger
logger = setup_logger()

def save_object(file_path, obj):
    """
    Saves a Python object (like a trained model or preprocessor) to a file using pickle.
    Args:
        file_path (str): Where to save the object.
        obj (object): The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error("Error saving object with pickle")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a pickled Python object from a file.
    Args:
        file_path (str): Path to the saved object.
    Returns:
        The loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.error("Error loading object with pickle")
        raise CustomException(e, sys)

def save_model(model, filepath):
    """
    Save a fitted model (scikit-learn/skorch) to disk using joblib.
    """
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error("Error saving model with joblib")
        raise CustomException(e, sys)

def load_model(filepath):
    """
    Load a saved model from disk using joblib.
    """
    try:
        return joblib.load(filepath)
    except Exception as e:
        logger.error("Error loading model with joblib")
        raise CustomException(e, sys)

def create_sequences(X, y, seq_len):
    """
    Converts flat tabular data into overlapping sliding window sequences
    for time-series models (LSTM, GRU, CNN-LSTM).
    Args:
        X: Features, shape (n_samples, n_features)
        y: Targets, shape (n_samples, 1) or (n_samples,)
        seq_len: Length of each sequence (window size)
    Returns:
        X_seq: shape (num_sequences, seq_len, n_features)
        y_seq: shape (num_sequences, 1)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

def evaluate_model(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Evaluates multiple models using GridSearchCV and returns their test R² scores.
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        models (dict): Model name → model object
        params (dict): Model name → hyperparameter grid
    Returns:
        report (dict): Model name → R² score
    """
    try:
        report = {}
        for model_name in models:
            model = models[model_name]
            param_grid = params.get(model_name, {})
            if param_grid:
                logger.info(f"Tuning {model_name} with GridSearchCV")
                grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                logger.info(f"Training {model_name} without hyperparameter tuning")
                model.fit(X_train, y_train)
                best_model = model
            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            logger.info(f"{model_name} test R²: {score:.4f}")
        return report
    except Exception as e:
        logger.error("Error during model evaluation in utils.py")
        raise CustomException(e, sys)
