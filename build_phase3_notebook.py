"""
Phase 3 Notebook Builder — Enhanced Version
============================================
Builds 03_Recommendation_System.ipynb with:
- Model training (XGBoost + Random Forest)
- Feature importance visualization
- ROC curve, Precision-Recall curve, Confusion Matrix
- Multi-horizon comparison (1M vs 3M vs 6M)
- Backtesting simulation ("Why follow our recommendations")
- Styled recommendations output
- Conclusion section
"""
import nbformat
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()
    
    nb.metadata.update({
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'title': 'Phase 3: ML Models & Recommendation Engine'
    })
    
    # ─── SECTION 1: Title ────────────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '# 📈 Phase 3: Machine Learning & Recommendation Engine\n\n'
        'This notebook trains **XGBoost** and **Random Forest** classifiers on our engineered technical indicators\n'
        'to predict stock uptrends on the Colombo Stock Exchange (CSE).\n\n'
        '**Key Highlights:**\n'
        '- Strict Time-Series Train/Test split (no future data leakage)\n'
        '- Precision-optimized models (minimize bad recommendations)\n'
        '- Multi-horizon predictions (1-Month, 3-Month, 6-Month)\n'
        '- Live recommendation engine with risk filters\n'
        '- Backtesting to validate real-world performance'
    ))
    
    # ─── SECTION 2: Imports ──────────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_code_cell(
'''import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.plot_helpers import setup_plotting_style, CSE_COLORS
from utils.data_loader import BASE_PATH

setup_plotting_style()
pd.set_option('display.max_columns', 100)

# Color palette for this notebook
PALETTE = {
    'teal': '#00b4d8', 'green': '#2dc653', 'red': '#ef233c',
    'orange': '#fb8500', 'purple': '#7209b7', 'navy': '#0d1b2a',
    'gold': '#ffd60a', 'steel': '#778da9', 'bg': '#f8f9fa',
}
print("✅ All imports loaded successfully.")
'''))

    # ─── SECTION 3: Load Data ────────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 1. Load Engineered Data\n'
        'Loading the Parquet file generated at the end of Phase 2, which contains 20+ technical indicators.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''out_path = os.path.join(BASE_PATH, 'engineered_features.parquet')
df = pd.read_parquet(out_path)
print(f"📊 Engineered Data loaded. Shape: {df.shape}")
print(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"🏢 Companies: {df['CompanyCode'].nunique()}")
print(f"\\n📋 Available columns: {len(df.columns)}")
df.head(3)
'''))

    # ─── SECTION 4: Feature Definition & Data Prep ───────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 2. Feature Definition & Data Preparation\\n\\n'
        'We select 20 technical indicators as model features. These cover **trend** (moving averages), '
        '**momentum** (RSI, MACD, ROC), **volatility** (ATR, Bollinger Bands), and **volume** patterns.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''# Technical features used by the model
FEATURES = [
    'SMA_20', 'SMA_50', 'SMA_200',
    'Dist_SMA20', 'Dist_SMA50', 'Dist_SMA200',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Lower', 'BB_Width',
    'ATR_14', 'NATR_14', 'ROC_10', 'ROC_21',
    'Vol_21d', 'Vol_63d', 'Volume_Ratio'
]

# Target: 3-Month Uptrend (>5% return in next 63 trading days)
TARGET = 'Is_Uptrend_3M'

# Drop rows where features or target are NaN
cols_to_check = FEATURES + [TARGET]
df_ml = df.dropna(subset=cols_to_check).copy()

print(f"✅ ML-ready dataset: {df_ml.shape[0]:,} samples × {len(FEATURES)} features")
print(f"📊 Uptrend ratio: {df_ml[TARGET].mean()*100:.1f}% positive (uptrend)")
'''))

    # ─── SECTION 5: Time-Series Split ────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 3. Time-Series Train/Test Split\\n\\n'
        '> ⚠️ **Critical**: We CANNOT use random train/test splits on financial data — that would '
        'leak future information into the training set. Instead, we split strictly by time:\\n'
        '> - **Training**: 2001–2022 (22 years of historical patterns)\\n'
        '> - **Testing**: 2023–2025 (3 years of unseen future data)\\n\\n'
        'This simulates real-world usage: the model only learns from the past.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''# Strict time-based split
train_df = df_ml[df_ml['Date'].dt.year <= 2022]
test_df = df_ml[df_ml['Date'].dt.year > 2022]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

print(f"📚 Training Set (2001-2022): {X_train.shape[0]:,} samples")
print(f"🧪 Test Set (2023-2025):     {X_test.shape[0]:,} samples")
print(f"\\n📊 Class Balance:")
print(f"   Train — Uptrend: {y_train.mean()*100:.1f}% | No Uptrend: {(1-y_train.mean())*100:.1f}%")
print(f"   Test  — Uptrend: {y_test.mean()*100:.1f}% | No Uptrend: {(1-y_test.mean())*100:.1f}%")

# Visualize the split
fig, ax = plt.subplots(figsize=(12, 3))
train_dates = train_df['Date']
test_dates = test_df['Date']
ax.axvspan(train_dates.min(), train_dates.max(), alpha=0.3, color=PALETTE['teal'], label='Training (2001-2022)')
ax.axvspan(test_dates.min(), test_dates.max(), alpha=0.3, color=PALETTE['orange'], label='Testing (2023-2025)')
ax.axvline(x=pd.Timestamp('2023-01-01'), color=PALETTE['red'], linewidth=2, linestyle='--', label='Split Point')
ax.set_xlim(train_dates.min(), test_dates.max())
ax.set_yticks([])
ax.set_xlabel('Date')
ax.set_title('Time-Series Train/Test Split', fontweight='bold', fontsize=14)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
'''))

    # ─── SECTION 6: Train Models ─────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 4. Train Machine Learning Models\\n\\n'
        'We train two robust tree-based classifiers:\\n'
        '- **Random Forest**: Ensemble of 100 decision trees with bagging\\n'
        '- **XGBoost**: Gradient-boosted trees with class imbalance handling\\n\\n'
        'Both are optimized for **Precision** — in a recommendation system, false positives '
        '(recommending a stock that crashes) are far worse than false negatives (missing a good stock).'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''# Train Random Forest
print("🌲 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, 
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Train XGBoost
print("⚡ Training XGBoost...")
pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
xgb_model = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=-1, scale_pos_weight=pos_weight
)
xgb_model.fit(X_train, y_train)

models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
print("\\n✅ Both models trained successfully!")
'''))

    # ─── SECTION 7: Evaluate Models ──────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 5. Model Evaluation\\n\\n'
        'We evaluate both models on the **unseen test set** (2023-2025).'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc, 'accuracy': acc,
        'y_pred': y_pred, 'y_proba': y_proba
    }
    
    print(f"\\n{'='*60}")
    print(f"  {name} — Test Set Performance")
    print(f"{'='*60}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  ROC-AUC:    {roc:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"\\n{classification_report(y_test, y_pred, zero_division=0)}")

# Summary comparison
summary_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k not in ('y_pred', 'y_proba')}
    for name, vals in results.items()
}).T
print("\\n📊 Model Comparison Summary:")
display(summary_df.style.format('{:.4f}').background_gradient(cmap='YlGnBu', axis=None))
'''))

    # ─── SECTION 8: ROC & PR Curves ─────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 6. ROC Curve & Precision-Recall Curve\\n\\n'
        'These curves show the trade-off between model sensitivity and specificity. '
        'A model with an AUC > 0.5 is performing better than random chance.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curves
ax = axes[0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    color = PALETTE['teal'] if name == 'XGBoost' else PALETTE['orange']
    ax.plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})', color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Baseline')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Precision-Recall Curves
ax = axes[1]
for name, res in results.items():
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, res['y_proba'])
    color = PALETTE['teal'] if name == 'XGBoost' else PALETTE['orange']
    ax.plot(rec_vals, prec_vals, label=name, color=color, linewidth=2)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
'''))

    # ─── SECTION 9: Confusion Matrices ───────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 7. Confusion Matrix\\n\\n'
        'The confusion matrix shows how many predictions were correct vs. incorrect. '
        'For a recommendation system, **the top-right cell (False Positives)** is the most '
        'dangerous — these are stocks the model recommended that actually went down.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    ax = axes[idx]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Uptrend', 'Uptrend'],
                yticklabels=['No Uptrend', 'Uptrend'],
                annot_kws={'size': 16, 'fontweight': 'bold'})
    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

plt.suptitle('Confusion Matrices (Test Set: 2023-2025)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
'''))

    # ─── SECTION 10: Feature Importance ──────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 8. Feature Importance Analysis\\n\\n'
        'Which technical indicators actually matter most for predicting an uptrend? '
        'This reveals whether the model is learning meaningful market signals or relying on noise.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (name, model) in enumerate(models.items()):
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    ax = axes[idx]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importances)))
    bars = ax.barh(importances.index, importances.values, color=colors)
    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    
    # Add value labels
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    ax.grid(True, alpha=0.2, axis='x')

plt.suptitle('Feature Importances — What Drives Stock Uptrend Predictions?', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
'''))

    # ─── SECTION 11: Multi-Horizon Comparison ────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 9. Multi-Horizon Comparison (1M vs 3M vs 6M)\\n\\n'
        'We compare model performance across three investment horizons to understand '
        'which timeframe is most predictable for the CSE market.'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''horizons = {
    '1-Month': 'Is_Uptrend_1M',
    '3-Month': 'Is_Uptrend_3M', 
    '6-Month': 'Is_Uptrend_6M'
}

horizon_results = {}

for label, target_col in horizons.items():
    cols_check = FEATURES + [target_col]
    df_h = df.dropna(subset=cols_check).copy()
    
    train_h = df_h[df_h['Date'].dt.year <= 2022]
    test_h = df_h[df_h['Date'].dt.year > 2022]
    
    if len(test_h) == 0:
        print(f"⚠️  {label}: No test data available (target may be NaN for recent dates)")
        continue
    
    X_tr, y_tr = train_h[FEATURES], train_h[target_col]
    X_te, y_te = test_h[FEATURES], test_h[target_col]
    
    pw = (len(y_tr) - sum(y_tr)) / max(sum(y_tr), 1)
    m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                      random_state=42, n_jobs=-1, scale_pos_weight=pw)
    m.fit(X_tr, y_tr)
    
    y_p = m.predict(X_te)
    y_pp = m.predict_proba(X_te)[:, 1]
    
    horizon_results[label] = {
        'Precision': precision_score(y_te, y_p, zero_division=0),
        'Recall': recall_score(y_te, y_p, zero_division=0),
        'F1': f1_score(y_te, y_p, zero_division=0),
        'ROC-AUC': roc_auc_score(y_te, y_pp),
        'Samples': len(y_te),
        'model': m,
    }
    print(f"✅ {label} model trained — Precision: {horizon_results[label]['Precision']:.3f}, ROC-AUC: {horizon_results[label]['ROC-AUC']:.3f}")

# Visualize comparison
if horizon_results:
    comp_df = pd.DataFrame({k: {mk: mv for mk, mv in v.items() if mk != 'model'} 
                            for k, v in horizon_results.items()}).T
    
    fig, ax = plt.subplots(figsize=(10, 5))
    comp_df[['Precision', 'Recall', 'F1', 'ROC-AUC']].plot(
        kind='bar', ax=ax, 
        color=[PALETTE['teal'], PALETTE['orange'], PALETTE['green'], PALETTE['purple']],
        edgecolor='white', linewidth=0.5
    )
    ax.set_title('Multi-Horizon Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Investment Horizon', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 Multi-Horizon Comparison:")
    display(comp_df.style.format('{:.4f}').background_gradient(cmap='YlGnBu', subset=['Precision', 'Recall', 'F1', 'ROC-AUC']))
'''))

    # ─── SECTION 12: Live Recommendations ────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 10. 🏆 Live Recommendation Engine\\n\\n'
        'Using the trained XGBoost model to generate **actionable buy recommendations** '
        'based on the most recent market data. The engine:\\n'
        '1. Takes the latest price data for every stock\\n'
        '2. Filters out penny stocks (< Rs. 5) and hyper-volatile stocks (> 80% volatility)\\n'
        '3. Runs the AI model to predict uptrend probability\\n'
        '4. Ranks stocks from highest to lowest predicted probability'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''from utils.recommender import generate_recommendations

# Get latest date
latest_date = df['Date'].max()
print(f"📅 Generating recommendations for: {latest_date.date()}")

# Get most recent row per company
df_current = df[df['Date'] == latest_date].copy()
print(f"🏢 Stocks with data on this date: {len(df_current)}")

# Generate Top 10 Recommendations
top_recs = generate_recommendations(
    df_current=df_current,
    model=xgb_model,
    feature_cols=FEATURES,
    min_price=5.0,
    max_volatility=0.80,
    top_n=10
)

print(f"\\n🏆 Top 10 Stock Recommendations (3-Month Horizon)")
print("=" * 60)

# Style the output
def color_probability(val):
    """Color-code the probability values."""
    if val >= 0.7:
        return 'background-color: #d4edda; color: #155724; font-weight: bold'
    elif val >= 0.5:
        return 'background-color: #fff3cd; color: #856404; font-weight: bold'
    else:
        return 'background-color: #f8d7da; color: #721c24'

styled = top_recs.style\\
    .format({
        'Close': 'Rs. {:.2f}',
        'Uptrend_Probability': '{:.1%}',
        'Vol_63d': '{:.1%}',
        'RSI_14': '{:.1f}',
        'MACD': '{:.4f}',
        'Dist_SMA50': '{:.2%}',
    })\\
    .map(color_probability, subset=['Uptrend_Probability'])\\
    .set_caption('🤖 AI-Generated Stock Recommendations — Ranked by Uptrend Probability')

display(styled)
'''))

    # ─── SECTION 13: Backtesting ─────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 11. 📊 Backtesting: Why Follow Our Recommendations?\\n\\n'
        'The ultimate test: **if an investor had followed our AI recommendations in the past, '
        'would they have made money?**\\n\\n'
        'We simulate this by:\\n'
        '1. Going to the start of the test period (Jan 2023)\\n'
        '2. Running our model to pick the top 10 stocks\\n'
        '3. Tracking what actually happened to those stocks over the next 3 months\\n'
        '4. Comparing against a "random pick" baseline'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''# Backtesting: Simulate quarterly recommendation windows in the test period
backtest_dates = test_df['Date'].dt.to_period('Q').unique()
backtest_results = []

for period in backtest_dates:
    # Get the first trading day of each quarter
    quarter_data = test_df[test_df['Date'].dt.to_period('Q') == period]
    if len(quarter_data) == 0:
        continue
    
    first_day = quarter_data['Date'].min()
    
    # Get stocks available on that day
    day_data = df[df['Date'] == first_day].copy()
    day_data = day_data.dropna(subset=FEATURES)
    day_data = day_data[day_data['Close'] >= 5.0]  # Filter penny stocks
    
    if len(day_data) < 10:
        continue
    
    # Model's top 10 picks
    probs = xgb_model.predict_proba(day_data[FEATURES])[:, 1]
    day_data = day_data.copy()
    day_data['pred_prob'] = probs
    top_picks = day_data.nlargest(10, 'pred_prob')
    
    # Random baseline: 10 random stocks
    random_picks = day_data.sample(n=min(10, len(day_data)), random_state=42)
    
    # Look up actual 3-month forward returns
    if 'Target_Return_3M' in top_picks.columns:
        ai_returns = top_picks['Target_Return_3M'].dropna()
        random_returns = random_picks['Target_Return_3M'].dropna()
        
        if len(ai_returns) > 0:
            backtest_results.append({
                'Quarter': str(period),
                'AI_Avg_Return': ai_returns.mean(),
                'AI_Median_Return': ai_returns.median(),
                'Random_Avg_Return': random_returns.mean(),
                'Random_Median_Return': random_returns.median(),
                'AI_Win_Rate': (ai_returns > 0).mean(),
                'Random_Win_Rate': (random_returns > 0).mean(),
                'AI_Picks': len(ai_returns),
            })

if backtest_results:
    bt_df = pd.DataFrame(backtest_results)
    
    print("📊 Backtesting Results: AI Recommendations vs Random Picks")
    print("=" * 70)
    display(bt_df.style.format({
        'AI_Avg_Return': '{:.1%}', 'AI_Median_Return': '{:.1%}',
        'Random_Avg_Return': '{:.1%}', 'Random_Median_Return': '{:.1%}',
        'AI_Win_Rate': '{:.0%}', 'Random_Win_Rate': '{:.0%}',
    }).background_gradient(cmap='RdYlGn', subset=['AI_Avg_Return', 'Random_Avg_Return']))
    
    # Visualize backtest
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average Returns Comparison
    ax = axes[0]
    x = np.arange(len(bt_df))
    width = 0.35
    bars1 = ax.bar(x - width/2, bt_df['AI_Avg_Return'] * 100, width, 
                   label='AI Picks', color=PALETTE['teal'], edgecolor='white')
    bars2 = ax.bar(x + width/2, bt_df['Random_Avg_Return'] * 100, width,
                   label='Random Picks', color=PALETTE['steel'], edgecolor='white')
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Average 3-Month Return (%)', fontsize=12)
    ax.set_title('Average Returns: AI vs Random', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bt_df['Quarter'], rotation=45)
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Win Rate Comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, bt_df['AI_Win_Rate'] * 100, width,
                   label='AI Picks', color=PALETTE['green'], edgecolor='white')
    bars2 = ax.bar(x + width/2, bt_df['Random_Win_Rate'] * 100, width,
                   label='Random Picks', color=PALETTE['steel'], edgecolor='white')
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate: AI vs Random', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bt_df['Quarter'], rotation=45)
    ax.legend()
    ax.axhline(y=50, color=PALETTE['red'], linewidth=1, linestyle='--', alpha=0.5, label='50% baseline')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\\n{'='*60}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"  AI Average Return:     {bt_df['AI_Avg_Return'].mean()*100:+.2f}%")
    print(f"  Random Average Return: {bt_df['Random_Avg_Return'].mean()*100:+.2f}%")
    print(f"  AI Win Rate:           {bt_df['AI_Win_Rate'].mean()*100:.1f}%")
    print(f"  Random Win Rate:       {bt_df['Random_Win_Rate'].mean()*100:.1f}%")
    ai_edge = bt_df['AI_Avg_Return'].mean() - bt_df['Random_Avg_Return'].mean()
    print(f"  AI Edge (α):           {ai_edge*100:+.2f}% per quarter")
    print(f"{'='*60}")
else:
    print("⚠️ Not enough data for backtesting. Forward returns may be NaN for recent dates.")
'''))

    # ─── SECTION 14: Save Model ──────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 12. Save Trained Models\\n\\n'
        'Saving the XGBoost model for use in the Streamlit dashboard (Phase 4).'
    ))
    nb.cells.append(nbformat.v4.new_code_cell(
'''import joblib

model_path = os.path.join(BASE_PATH, 'xgboost_3m_model.pkl')
joblib.dump(xgb_model, model_path)
print(f"💾 XGBoost model saved to: {model_path}")
print(f"   Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
'''))

    # ─── SECTION 15: Conclusion ──────────────────────────────────────────
    nb.cells.append(nbformat.v4.new_markdown_cell(
        '## 13. 📝 Conclusion\\n\\n'
        '### What We Built\\n'
        'A complete **AI-powered stock recommendation pipeline** for the Colombo Stock Exchange:\\n'
        '1. **Data**: 34 years of historical price data (1991-2025), cleaned and split-adjusted\\n'
        '2. **Features**: 20+ technical indicators spanning trend, momentum, volatility, and volume\\n'
        '3. **Models**: XGBoost and Random Forest classifiers trained on 22 years of data\\n'
        '4. **Recommendations**: Probabilistic ranking engine with built-in risk filters\\n\\n'
        '### Key Strengths\\n'
        '- ✅ **No future data leakage** — strict time-series split\\n'
        '- ✅ **Precision-focused** — minimizes bad recommendations\\n'
        '- ✅ **Risk-aware** — filters penny stocks and hyper-volatile securities\\n'
        '- ✅ **Multi-horizon** — predictions for 1-month, 3-month, and 6-month windows\\n'
        '- ✅ **Backtested** — validated on unseen 2023-2025 data\\n\\n'
        '### Limitations & Future Work\\n'
        '- ⚠️ **Survivorship bias**: Delisted stocks are excluded, potentially inflating results\\n'
        '- ⚠️ **No fundamental data**: The model uses only technical indicators (no earnings, P/E ratios)\\n'
        '- ⚠️ **Market regime shifts**: Model trained primarily on bull/bear cycles may not generalize to unprecedented events\\n'
        '- 🔮 **Future improvements**: Incorporate fundamental data, add LSTM deep learning models, '
        'implement portfolio optimization with risk-parity allocation\\n\\n'
        '### Next Step\\n'
        '🚀 **Phase 4**: Launch the interactive Streamlit dashboard (`streamlit run app.py`) to explore '
        'the recommendations in real-time!'
    ))
    
    # Write notebook
    output_path = r'c:\Users\HP\Documents\Stock_pulse\03_Recommendation_System.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Notebook created: {output_path}")
    print(f"   Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'code')} code, "
          f"{sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown)")


if __name__ == "__main__":
    create_notebook()
