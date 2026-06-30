import nbformat
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()
    
    # Metadata
    nb.metadata.update({
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'title': 'Phase 3: ML Models & Recommendation Engine'
    })
    
    # Setup
    nb.cells.append(nbformat.v4.new_markdown_cell('# Phase 3: Machine Learning & Recommendation Engine\n\nTraining XGBoost and Random Forest models on our engineered features to predict stock uptrends, and building the final recommendation system.'))
    
    nb.cells.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

from utils.plot_helpers import setup_plotting_style, CSE_COLORS
from utils.data_loader import BASE_PATH
from utils.model_trainer import prepare_ml_dataset, train_models, evaluate_models
from utils.recommender import generate_recommendations

setup_plotting_style()
pd.set_option('display.max_columns', 100)
'''))

    # Load Engineered Data
    nb.cells.append(nbformat.v4.new_markdown_cell('## 1. Load Engineered Data\nLoading the Parquet file generated at the end of Phase 2.'))
    nb.cells.append(nbformat.v4.new_code_cell('''out_path = os.path.join(BASE_PATH, 'engineered_features.parquet')
df = pd.read_parquet(out_path)
print(f"Engineered Data loaded. Shape: {df.shape}")
'''))

    # Data Preparation (Time-Series Split)
    nb.cells.append(nbformat.v4.new_markdown_cell('## 2. Train-Test Split (Time-Series)\nBecause this is financial data, we split strictly by time to avoid future-leakage. Train: 2001-2022, Test: 2023-2025.'))
    nb.cells.append(nbformat.v4.new_code_cell('''# We will train models for the 3-Month (Medium Term) Horizon first
target = 'Is_Uptrend_3M'

X_train, y_train, X_test, y_test, train_df, test_df, features = prepare_ml_dataset(df, target_col=target)

print(f"Training Set (2001-2022): {X_train.shape[0]} samples")
print(f"Testing Set (2023-2025): {X_test.shape[0]} samples")
print(f"Class Balance (Train): {y_train.mean()*100:.2f}% Uptrends")
'''))

    # Train Models
    nb.cells.append(nbformat.v4.new_markdown_cell('## 3. Train Machine Learning Models\nTraining Random Forest and XGBoost classifiers optimized for Precision.'))
    nb.cells.append(nbformat.v4.new_code_cell('''models = train_models(X_train, y_train)
results = evaluate_models(models, X_test, y_test)
'''))

    # Feature Importance
    nb.cells.append(nbformat.v4.new_markdown_cell('## 4. Feature Importance Analysis\nWhat technical indicators actually matter most for predicting an uptrend?'))
    nb.cells.append(nbformat.v4.new_code_cell('''xgb_model = models['XGBoost']

# Plot Feature Importances
importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title('XGBoost Feature Importances (Predicting 3M Uptrends)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
'''))

    # The Recommendation Engine
    nb.cells.append(nbformat.v4.new_markdown_cell('## 5. Live Recommendation Engine\nExtracting the latest available date for each stock and feeding it into the trained XGBoost model to get actionable buy recommendations.'))
    nb.cells.append(nbformat.v4.new_code_cell('''# 1. Get the most recent date in the dataset
latest_date = df['Date'].max()
print(f"Generating recommendations for market data as of: {latest_date.date()}")

# 2. Extract the last row for each company
df_current = df[df['Date'] == latest_date].copy()

# 3. Generate Top 10 Recommendations
# We filter out penny stocks (< Rs. 5) and highly volatile stocks (> 80% annualized volatility)
top_recs = generate_recommendations(
    df_current=df_current,
    model=xgb_model,
    feature_cols=features,
    min_price=5.0,
    max_volatility=0.80,
    top_n=10
)

display(top_recs)
'''))

    # Save Models
    nb.cells.append(nbformat.v4.new_markdown_cell('## 6. Save Models\nSaving the XGBoost model for future inference without retraining.'))
    nb.cells.append(nbformat.v4.new_code_cell('''import joblib
model_path = os.path.join(BASE_PATH, 'xgboost_3m_model.pkl')
joblib.dump(xgb_model, model_path)
print(f"Model saved to {model_path}")
'''))

    # Write notebook to file
    with open('c:/Users/HP/Documents/Stock_pulse/03_Recommendation_System.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print("Notebook 03_Recommendation_System.ipynb created successfully.")

if __name__ == "__main__":
    create_notebook()
