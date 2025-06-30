
"""
Main workflow runner for the Appliance Energy Prediction Pipeline
Steps:
1. Preprocess & feature engineer
2. Train model
3. Evaluate & print results
"""

import numpy as np
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import setup_logger
from src.exception import CustomException

logger = setup_logger()

def main():
    try:
        logger.info("=== Appliance Energy Prediction Pipeline Started ===")

        # ===1. Preprocessing  ===
        logger.info("Step 1: Running data transformation...")
        transformer = DataTransformation()
        status = transformer.run(
            raw_path="data/raw/energy_data_set.csv"
        )

        logger.info("Data preprocessing completed successfully.")

        # === 2. Model Training Step ===
        logger.info("Step 2: Running model training and evaluation...")

        # Load preprocessed arrays
        train_arr = np.load("data/processed/train_arr.npy")
        test_arr = np.load("data/processed/test_arr.npy")

        # Initialize model trainer
        model_trainer = ModelTrainer()

        # Pass arrays as arguments to initiate_model_trainer
        final_r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logger.info(f"Pipeline finished successfully! Final test R² score: {final_r2:.4f}")
        print(f"Final test R² score: {final_r2:.4f}")

    except CustomException as ce:
        logger.error(f"CustomException in pipeline: {ce}")
        print(f"CustomException: {ce}")
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

     












 