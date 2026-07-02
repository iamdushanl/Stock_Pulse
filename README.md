# Stock Pulse — CSE Market Analysis & Share Recommendation System

A professional data science project analyzing historical data from the Colombo Stock Exchange (CSE) and building a machine-learning-powered share recommendation system.

## Project Structure

```
Stock_Pulse/
├── 01_CSE_Exploratory_Data_Analysis.ipynb   # Phase 1: EDA Notebook
├── 02_Feature_Engineering_and_Targets.ipynb # Phase 2: Feature Engineering
├── 03_Recommendation_System.ipynb           # Phase 3: ML & Recommendations
├── utils/
│   ├── __init__.py              # Package exports
│   ├── data_loader.py           # Multi-era data loading from ZIPs
│   ├── data_cleaning.py         # Split adjustments & calendar alignment
│   ├── features.py              # Technical indicators (RSI, MACD, etc.)
│   ├── targets.py               # Forward return target definitions
│   ├── model_trainer.py         # XGBoost & Random Forest training
│   ├── recommender.py           # Stock recommendation engine
│   └── plot_helpers.py          # Publication-quality plot utilities
├── Dataset/                     # Raw CSE data (not tracked in git)
└── notebook_parts/              # Notebook assembly scripts
```

## Phases

### Phase 1 — Exploratory Data Analysis
- Parses 34 years of CSE data (1991–2025) from yearly Excel/CSV files inside ZIP archives.
- Handles two distinct schema eras with robust error handling.
- 16-section professional notebook covering distributions, correlations, volatility, and trend analysis.

### Phase 2 — Feature Engineering & Target Definition
- Stock split backward adjustments using official CSE corporate actions data.
- Technical indicators: SMA, MACD, RSI, Bollinger Bands, ATR, Stochastic Oscillator.
- Target variables: 1M, 3M, 6M forward returns with binary uptrend classification labels.

### Phase 3 — Machine Learning & Recommendation System
- Time-series train/test split (2001–2022 train, 2023–2025 test) to prevent data leakage.
- XGBoost and Random Forest classifiers optimized for precision.
- Live recommendation engine that ranks stocks by predicted uptrend probability.

## Setup

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scipy statsmodels xgboost scikit-learn tqdm openpyxl xlrd nbformat joblib

# Run notebooks in order
jupyter notebook
```

## License
University final-year project — Colombo Stock Exchange data used under academic license.
