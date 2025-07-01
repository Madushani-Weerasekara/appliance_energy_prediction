"""
ModelTrainer for Appliance Energy Prediction
-------------------------------------------
Steps:
- Loads preprocessed train/test arrays and selected feature list
- Trains a regression model (LinearRegression & RandomForestRegressor)
- Evaluates model performance on the test set
- Saves trained model  
- Logs every step for traceability and debugging

"""

import os  # For directory/file operations
import sys
import numpy as np # For fast array operations (used for data storage and loading)
from dataclasses import dataclass

# Model libraries
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logger
from src.exception import CustomException      # Project-wide exception handling

# Utilities (custom)
from src.utils import save_object, load_object, evaluate_model



@dataclass
class ModelTrainerConfig:
    """
    Configuration for saving trained model and loading data.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    train_data_path: str = os.path.join("artifacts", "train.npy")
    test_data_path: str = os.path.join("artifacts", "test.npy")
   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

     
    def initiate_model_trainer(self, train_arr, test_arr):
        """
        Train multiple models and select the best one based on R² score.
        
        Args:
            train_array (np.ndarray): Combined training data with last column as target.
            test_array (np.ndarray): Combined testing data with last column as target.

        Returns:
            float: R² score of the best model on test data.
        """
        try:
            logger.info("Unpacking features and target from train/test arrays (already split previously).")

            # Unpacking: Features = all columns except last, Target = last column
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # Define base models
            logger.info("Defining baseline regression models")
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor()
            }

            logger.info("Defining hyperparameter grid for each model")
            params = {
                "Linear Regression": {},

                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },

                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.3]
                },

                "Decision Tree": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                },

                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                },

                "Support Vector Regressor": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                }
            }
            
            # Evaluate all models (using utils.py function)
            logger.info("Starting model evaluation using utils.evaluate_model")
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logger.info(f"Best Model: {best_model_name} | R² Score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No model met minimum performance threshold (R² < 0.6)", sys)


            logger.info("Saving best model to disk")
            save_object(
                file_path="artifacts/best_ml_model.pkl",  # More explicit name!
                obj=best_model
            )

            predictions = best_model.predict(X_test)
            final_r2 = r2_score(y_test, predictions)

            logger.info(f"Final R² Score on test set: {final_r2:.4f}")
            return final_r2, best_model


        except Exception as e:
            logger.error("Error in initiate_model_trainer")
            raise CustomException(e, sys)
