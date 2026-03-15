import pandas as pd
import numpy as np
import os
import joblib
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.feature_engineering import engineer_features, get_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved')


def load_model(model_name: str, target: str):
    path = os.path.join(MODELS_DIR, f'{model_name}_{target}.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def load_scaler(target: str):
    path = os.path.join(MODELS_DIR, f'scaler_{target}.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Fill all NaN values so every model can run safely."""
    X = X.copy()
    for col in X.columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)
    return X


def predict_race(race_df: pd.DataFrame, model_name: str = 'XGBoost') -> pd.DataFrame:
    features = get_feature_columns()
    race_df = engineer_features(race_df)
    X = clean_features(race_df[features])  # ← NaN fix applied here

    results = race_df[['Abbreviation', 'FullName', 'TeamName',
                        'GridPosition', 'QualiPosition']].copy()

    for target in ['Winner', 'Podium', 'DNF']:
        model = load_model(model_name, target)
        if model is None:
            print(f"Model not found: {model_name}_{target}")
            results[f'{target}_Prob'] = 0.0
            continue

        try:
            if model_name == 'NeuralNet':
                scaler = load_scaler(target)
                X_in = scaler.transform(X) if scaler else X.values
            else:
                X_in = X

            proba = model.predict_proba(X_in)[:, 1]
            results[f'{target}_Prob'] = (proba * 100).round(1)
        except Exception as e:
            print(f"Prediction error for {model_name}/{target}: {e}")
            results[f'{target}_Prob'] = 0.0

    results = results.sort_values('Winner_Prob', ascending=False).reset_index(drop=True)
    results['PredictedPosition'] = results.index + 1
    return results


def get_model_comparison() -> pd.DataFrame:
    path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()