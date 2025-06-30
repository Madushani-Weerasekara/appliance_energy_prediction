"""
DataTransformer class for complete preprocessing and feature engineering
Pipeline includes:
- Data loading & cleaning
- Handling missing values and outliers
- Scaling
- Time-based features
- Lag and rolling features
- Holiday & season features
- Train-test split

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from dataclasses import dataclass
import pickle
import holidays
import os
import sys

from src.logger import setup_logger
from src.exception import CustomException
from sklearn.ensemble import RandomForestRegressor
  


logger = setup_logger()
 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    selected_features_path: str = os.path.join('artifacts', 'selected_features.pkl')
    processed_train_arr_path: str = os.path.join('data', 'processed', 'train_arr.npy')
    processed_test_arr_path: str = os.path.join('data', 'processed', 'test_arr.npy')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def check_nulls(self, df):
        null_sum = df.isnull().sum()
        total_nulls = null_sum.sum()
        if total_nulls > 0:
            message = f"Null values found! Total: {total_nulls}.\n{null_sum[null_sum>0]}"
            logger.error(message)
            raise CustomException(message)
        else:
            logger.info("No null values found.")

    def preprocess_and_engineer(self, df):
        """
        Complete preprocessing and feature engineering on raw DataFrame.
        """
        try:
            logger.info("Starting preprocessing and feature engineering...")

            self.check_nulls(df)

            # 1. Handle missing values
            df.ffill(inplace=True)
            logger.info("Forward-filled missing values.")

            # 2. Cap outliers in 'Appliances'
            q1 = df['Appliances'].quantile(0.25)
            q3 = df['Appliances'].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            df['Appliances_capped'] = df['Appliances'].clip(lower=lower, upper=upper)
            logger.info(f"Outliers capped at [{lower}, {upper}].")

            # 3. Feature engineering: time, domain, lag, rolling
            df['hour'] = df.index.hour
            df['month'] = df.index.month
            df['day_of_week'] = df.index.dayofweek
            df['NSM'] = df.index.hour*3600 + df.index.minute*60 + df.index.second
            df['WeekStatus'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # US season mapping
            def month_to_season_us(month):
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8]:
                    return 'summer'
                elif month in [9, 10, 11]:
                    return 'fall'

            df['season'] = df['month'].apply(month_to_season_us)

            # Add holiday indicator using US holidays
            us_holidays = holidays.US(years=df.index.year.unique())
            df['is_holiday'] = df.index.normalize().map(lambda x: x.date()).isin(us_holidays).astype(int)
             
            # Lag and rolling
            df['Appliances_capped_lag1'] = df['Appliances_capped'].shift(1)
            df['Appliances_capped_lag3'] = df['Appliances_capped'].shift(3)

            df['Appliances_capped_rollmean1h'] = df['Appliances_capped'].rolling(window=6).mean()
            df['Appliances_capped_rollmean3h'] = df['Appliances_capped'].rolling(window=18).mean()

            # Interaction features 
            #   T1 x RH_1, T2 x RH_2, T_out x RH_out
            interaction_pairs = [
                ('T1', 'RH_1'), ('T2', 'RH_2'), ('T_out', 'RH_out'),
                ]

            # Add more as desired, e.g. ('T3', 'RH_3'), etc.
            
            for t, rh in interaction_pairs:
                if t in df.columns and rh in df.columns:
                    df[f'{t}_x_{rh}'] = df[t] * df[rh]
                    logger.info(f"Created interaction feature: {t}_x_{rh}")
            # Drop rows with NA due to rolling/lags
            df.dropna(inplace=True)
            logger.info("Feature engineering complete.")

            if 'Appliances' in df.columns:
                df = df.drop(columns=['Appliances'])
                logger.info("Removed 'Appliances' column to prevent data leakage.")

            # Encode categoricals
            df = self.encode_categoricals(df)
            logger.info("Categorical encoding complete.")

            self.check_nulls(df)
            return df
        
        except Exception as e:
            logger.error(f"Error in preprocess_and_engineer: {e}")
            raise CustomException(f"Error in preprocess_and_engineer: {e}", error=e)
        
    def encode_categoricals(self, df):
        """
        Directly encode categorical columns into numerical codes by mapping values.
        - WeekStatus: 'weekday' -> 0, 'weekend' -> 1
        - season: 'winter' -> 0, 'spring' -> 1, 'summer' -> 2, 'fall' -> 3
        - is_holiday: already 0/1, so no change needed
        """
        # Convert to int (after confirming no NaN)
        df['WeekStatus'] = df['WeekStatus'].astype(int)

        # season: map to 0-3
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        df['season'] = df['season'].map(season_map)

        # is_holiday: already 0/1, but you can make sure:
        df['is_holiday'] = df['is_holiday'].astype(int)
        return df
    
    def correlation_selection(self, df, target_col, threshold=0.9):
        """
        Remove one of each pair of highly correlated features.
        """
        try: 
            logger.info("Running correlation-based feature selection.")
            corr_matrix = df.drop(columns=[target_col]).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            selected = [col for col in df.columns if col not in to_drop + [target_col]]
            logger.info(f"Correlation selection dropped {len(to_drop)} features: {to_drop}")
            logger.info(f"Correlation selection kept {len(selected)} features.")
            return selected
        
        except Exception as e:
            logger.error(f"Error in correlation_selection: {e}")
            raise CustomException(f"Error in correlation_selection: {e}", error=e)


    def feature_selection_tree(self, train_df, selected, target_col, num_features=15):
        """
        Use RandomForest feature importance for selection.
        """
        try:
            logger.info("Running tree-based feature selection.")
            X = train_df[selected]
            y = train_df[target_col]
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:num_features]
            final_features = [X.columns[i] for i in indices]
            logger.info(f"Tree-based selection kept {len(final_features)} features: {final_features}")
         
            return final_features
    
        except Exception as e:
            logger.error(f"Error in feature_selection_tree: {e}")
            raise CustomException(f"Error in feature_selection_tree: {e}", error=e)

    def split_by_time(self, df, train_frac=0.8):
        """
        Split data into train/test by time order (no shuffling).
        """
        try:
            n = len(df)
            split_idx = int(n * train_frac)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            logger.info(f"Data split by time: {len(train_df)} train, {len(test_df)} test rows.")
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error in split_by_time: {e}")
            raise CustomException(f"Error in split_by_time: {e}", error=e)
        
    def run(self, raw_path, target_column='Appliances_capped', num_features=15):
        """
        Complete all steps: preprocess, feature engineer, split, feature select, scale, save.
        """
        try:
            # 1. Load data
            df = pd.read_csv(raw_path, parse_dates=['date'])
            df.set_index('date', inplace=True)
            logger.info(f"Loaded data from {raw_path}. Shape: {df.shape}")

            # 2. Preprocess and feature engineer
            df = self.preprocess_and_engineer(df)

            # 3. Split by time
            train_df, test_df = self.split_by_time(df, train_frac=0.8)

            # 4. Feature selection (correlation, then tree-based) on training set only
            selected_corr = self.correlation_selection(train_df, target_col=target_column)
            selected_tree = self.feature_selection_tree(train_df, selected_corr, target_col=target_column, num_features=num_features)
            logger.info(f"Final features: {selected_tree}")

             # 5. Final feature data (drop date column if exists)
            if 'date' in train_df.columns:
                train_df = train_df.drop(columns=['date'])
            if 'date' in test_df.columns:
                test_df = test_df.drop(columns=['date'])

            # 6. Final feature data
            X_train = train_df[selected_tree]
            X_test = test_df[selected_tree]
            y_train = train_df[target_column]
            y_test = test_df[target_column]

            # 7. Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 8. Save scaler
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved scaler to {self.config.preprocessor_obj_file_path}")

            # 9. Save processed arrays
            os.makedirs(os.path.dirname(self.config.processed_train_arr_path), exist_ok=True)
            train_arr = np.c_[X_train_scaled, y_train.values]
            test_arr = np.c_[X_test_scaled, y_test.values]
            np.save(self.config.processed_train_arr_path, train_arr)
            np.save(self.config.processed_test_arr_path, test_arr)
            logger.info(f"Saved processed train array to {self.config.processed_train_arr_path}")
            logger.info(f"Saved processed test array to {self.config.processed_test_arr_path}")
            logger.info(f"Train array shape: {train_arr.shape}, Test array shape: {test_arr.shape}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path, self.config.selected_features_path

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise CustomException(f"Pipeline error: {e}", error=e)
        


