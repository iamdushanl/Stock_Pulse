import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, f1_score, roc_auc_score

def prepare_ml_dataset(df, target_col='Is_Uptrend_3M', feature_cols=None):
    """
    Cleans the dataset by dropping NaNs and returns X and y for modeling.
    """
    if feature_cols is None:
        # Default technical features
        feature_cols = [
            'SMA_20', 'SMA_50', 'SMA_200',
            'Dist_SMA20', 'Dist_SMA50', 'Dist_SMA200',
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Width',
            'ATR_14', 'NATR_14', 'ROC_10', 'ROC_21',
            'Vol_21d', 'Vol_63d', 'Volume_Ratio'
        ]
    
    # We must drop rows where either the features or the target are NaN
    cols_to_check = feature_cols + [target_col]
    df_clean = df.dropna(subset=cols_to_check).copy()
    
    # Time-based train/test split (Train: 2001-2022, Test: 2023-2025)
    train_df = df_clean[df_clean['Date'].dt.year <= 2022]
    test_df = df_clean[df_clean['Date'].dt.year > 2022]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test, train_df, test_df, feature_cols


def train_models(X_train, y_train):
    """
    Trains XGBoost and Random Forest classifiers.
    """
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train) # Handle imbalance
    )
    xgb_model.fit(X_train, y_train)
    
    return {'RandomForest': rf_model, 'XGBoost': xgb_model}


def evaluate_models(models, X_test, y_test):
    """
    Evaluates models focusing on Precision and ROC-AUC.
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            roc_auc = float('nan')
        
        print(f"\n--- {name} Performance ---")
        print(f"Precision (Uptrend): {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        results[name] = {
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
    return results
