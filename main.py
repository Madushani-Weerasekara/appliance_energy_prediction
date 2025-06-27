"""
Main pipeline script to run the full workflow for appliance energy prediction.
Calls modular OOP classes for preprocessing, feature engineering, etc.

"""

import pandas as pd
from src.components.data_preprocessing import DataPreprocessor

# STEP 1: Data Preprocessing
print("\n=== Step 1: Data Preprocessing ===")
preprocessor = DataPreprocessor("data/raw/energy_data_set.csv")
preprocessor.run_all("data/processed/processed_energy_data_set.csv")







 