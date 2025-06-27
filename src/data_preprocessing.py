# Nodule for loading and cleaning the raw data
# Handle missing values
# Remove/Impute outliers
# Normalize features(MnMax/Standard)
# Split data into train/test(by time)

import pandas as pd
import numpy as np
from sklearn.preprocessing  import MinMaxScaler
import os

class DataPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the preprocessor and loads the data from CSV.
        Args:
            file_path(str): path to the csv file
        """
        self.file_path = file_path
        self.df = self.load_data()

    def load_data(self):
        """
        Loads the datset and set the date column as index.
        Returns:
            pd.DataFrame:  Loaded DataFrame

        """
        df = pd.read_csv(self.file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print("Data loaded. Shape:", df.shape)
        return df
    
    def detect_missing_values(self):
        # Prints  & return the number of missing values per column.
        missing_values = self.df.isnull().sum()
        print("\nMissing values per column:\n", missing_values)
        return missing_values
    
    def handle_missing_values(self):
        # Handles missing values using forward fill('ffill) which is best for time series.
        print("\nHandling missing values using forward fill('ffill'):")
        self.df.fillna(method='ffill', inplace=True)
        print("Missing value handling complete.")
        self.detect_missing_values()

    










# Example Usage/Test code(Remove these lines when importing in main code)

if __name__ == "__main__":
    processor = DataPreprocessor("../data/raw/energy_data_set.csv")
    #print(processor.df.columns)
    #print(processor.df.info())
    processor.handle_missing_values()
    print(processor.df.head(3))
 