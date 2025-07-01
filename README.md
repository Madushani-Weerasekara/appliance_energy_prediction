# Appliance Energy Prediction

This project predicts household appliance energy consumption using deep learning (LSTM, GRU, CNN-LSTM) and a robust machine learning pipeline. It includes data preprocessing, feature engineering, model training, hyperparameter tuning (Optuna), and visualization.

---

## Overview

- _Preprocessing:_ Cleans data, adds engineered features (lags, rolling means, holidays, seasons), and selects best features.
- _Modeling:_ Implements LSTM, GRU, and CNN-LSTM regression models in PyTorch.
- _Hyperparameter Tuning:_ Uses Optuna to find the best model settings.
- _Evaluation:_ Reports MSE, RMSE, MAE, R², and plots real vs. predicted values.

---

## Setup Instructions

### 1. _Clone the repository_

```bash
git clone https://github.com/yourusername/appliance_energy_prediction.git
cd appliance_energy_prediction
```

Data

Download the Appliance Energy Prediction Data Set (UCI)

Place the raw file as data/raw/energy_data_set.csv
