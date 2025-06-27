"""
DataPreprocessor class for loading, cleaning, and scaling the energy dataset.
- Handles missing values (forward fill for time series)
- Caps outliers in the target variable ('Appliances') using the IQR method
- Scales all numeric features (except target and capped target) using StandardScaler
- Saves the processed DataFrame to the data/raw/ directory

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing  import StandardScaler
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
        self.df.ffill(inplace=True)
        print("Missing value handling complete.")
        self.detect_missing_values()

    def handle_outliers(self):
        """
        Caps outliers in the specified column using the IQR method.
        Creates a new column '<column>'_capped

        """
        q1 = self.df['Appliances'].quantile(0.25)
        q3 = self.df['Appliances'].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        before_min = self.df['Appliances'].min()  # Min before capping outliers
        before_max = self.df['Appliances'].max()  # Max before capping outliers

        self.df['Appliances_capped']  = self.df['Appliances'].clip(lower=lower_bound, upper=upper_bound)

        after_min = self.df['Appliances_capped'].min() # Min after capped the outliers
        after_max = self.df['Appliances_capped'].max() # Max after capped the outliers

        print(f"\nCapped 'Appliances_capped' outliers at [{lower_bound:.2f}, {upper_bound:.2f}].")
        print(f"\nBefore capping: min={before_min}, max={before_max}")
        print(f"\nAfter capping: min={after_min}, max={after_max}")

    def scale_features(self):
        """
        Scale specific columns using StandardScaler.
        Creates new columns with '_scaled' suffix.

        """
        # Select all numeric columns except 'Appliances' and 'Appliances_capped'
        exclude_cols = ['Appliances', 'Appliances_capped']
        numeric_cols = [
            col for col in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[col]) and col not in exclude_cols
        ]

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(self.df[numeric_cols])

        for i, col in enumerate(numeric_cols):
            self.df[col  + '_scaled'] = scaled_values[:, i]
        print(f"\nScaled columns: {numeric_cols} (zero mean unit variance)")

    def save_processed_data(self, save_path):
        """
        Save the processed DataFrame to given path as a CSV file.
        Args:
            save_path(str): Path(including filename)  to save the processed CSV.

        """

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path, exist_ok=True))
        self.df.to_csv(save_path)

    def get_data(self):
        # Returns the  proessed DataFrame

        return self.df

def run_all(self, save_path):
        """Industry-style pipeline method."""
        print("Starting full data preprocessing pipeline...")
        self.detect_missing_values()
        self.handle_missing_values()
        self.handle_outliers()
        self.scale_features()
        self.save_processed_data(save_path)
        print("Preprocessing pipeline complete! Processed data saved to:", save_path)


"""
# Example Usage/Test code(Remove these lines when importing in main code)

if __name__ == "__main__":
    processor = DataPreprocessor("../data/raw/energy_data_set.csv")
    #print(processor.df.columns)
    #print(processor.df.info())
    processor.detect_missing_values()
    processor.handle_missing_values()
    processor.handle_outliers()
    processor.scale_features()
    print(processor.df.head(3))
    df_clean = processor.get_data()
 
    """