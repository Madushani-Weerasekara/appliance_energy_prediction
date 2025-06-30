
"""
Main workflow runner for the Appliance Energy Prediction Pipeline
Steps:
1. Preprocess & feature engineer
2. Train model
3. Evaluate & print results
"""

import numpy as np
from src.components.data_transformation import DataTransformation
from src.logger import setup_logger
from src.exception import CustomException

logger = setup_logger()
logger.info("Starting main pipeline...")

# 1. Preprocessing
transformer = DataTransformation()
status = transformer.run(
    raw_path="data/raw/energy_data_set.csv"
)

logger.info("Data preprocessing complete.")









 