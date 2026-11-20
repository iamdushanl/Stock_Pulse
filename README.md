# 📈 Stock Pulse

**An explainable, multi-horizon stock research and recommendation system for the Colombo Stock Exchange (CSE).**

Stock Pulse is a university data science project combining historical CSE market data, technical analysis, financial fundamentals, corporate disclosures, and machine learning to research investment opportunities across three strategies.

> Research software only. Historical backtests and model outputs are not guarantees of future performance or personalized investment advice.

## Investment engines

| Engine | Horizon | Objective | Current status |
|---|---:|---|---|
| Short-term trading | 0–3 months | Technical momentum, liquidity, volatility and event-aware ranking | Phase 4.1 candidates benchmarked; independent verification pending |
| Medium-term growth | 3–12 months | Earnings growth, profitability, valuation and momentum | Blocked pending historical point-in-time fundamentals |
| Long-term wealth creation | 1–5+ years | Business quality, valuation, dividends and total shareholder return | Blocked pending multi-year fundamentals and dividend-reinvested return data |

## Project progress

| Phase | Status | Summary |
|---|---|---|
| Phase 1 – EDA | Complete | Exploratory analysis of historical CSE market and security data |
| Phase 2 – Data enrichment | Complete with coverage limitations | Historical market sources and latest available CSE financial disclosures collected and extracted |
| Phase 2C/2D – Financial data quality | Reported complete / follow-up validation | Financial facts, period classification, reconciliation and source traceability |
| Phase 3 – Feature engineering | Generated | Separate technical, medium-term and long-term feature matrices |
| Phase 3.5 – Feature audit | Reported complete | Identified point-in-time leakage in static 2026 fundamentals and announcement flags; short-term engine approved with limitations |
| Phase 4.1 – Short-term model benchmarking | Reported complete | Ridge, Random Forest, XGBoost and LightGBM compared across 5-, 20- and 60-day targets |
| Phase 4.2 – Independent backtest verification | Complete | Verified backtests, audited transaction fees (2.24% round-trip), isolated COMB bad tick inflation, tested T+1 execution realism; LightGBM 20d verified with limitations |
| Phase 5 – Recommendation application | Planned | Integrate validated models, explanations, risk measures and recommendation history |

**Current modelling scope:** The reported Phase 4.1 historical panel covers 20 S&P SL20 constituents, not the full CSE universe. The reported LightGBM 20-trading-day model is a provisional research candidate until independent backtest verification is complete.

## Phase 4.1 & 4.2 short-term model research & verification

The supplied Phase 4.1 report benchmarked these horizon-specific candidates:

| Target | Candidate | Phase 4.1 Reported CAGR | Phase 4.2 Cleaned Realistic CAGR | Net Sharpe | Verification Status | Verdict |
|---|---|:---:|:---:|:---:|---|---|
| 5 trading days | Random Forest | 426.85% | 64.72% | 1.87 | NOT VERIFIED | Rejected: 23% fee drag; inflated by weekly compounding on bad tick bounce-backs. |
| **20 trading days** | **LightGBM** | **223.22%** | **55.55%** | **1.21** | **VERIFIED WITH LIMITATIONS** | **Primary Candidate:** Genuine rank alpha ($p < 10^{-15}$); beats ASPI (46.67%). |
| 60 trading days | Ridge Regression | 79.30% | 77.86% | 1.67 | VERIFIED WITH LIMITATIONS | Defensive overlay: Low turnover (4.6% annual drag), near-zero drawdown (-3.3%). |

### Phase 4.2 Independent Verification Findings
- **Root-Cause of Return Inflation:** Phase 4.1's 223.22% LightGBM CAGR was inflated by an unadjusted clerical data anomaly on Commercial Bank (`COMB`) in August 2024 (prices dropped to Rs. 3.40 and reverted to Rs. 89.70, yielding an artificial $+2361\%$ gain in month 8) and a 50% under-deduction of statutory round-trip transaction fees (deducting 1.12% on one-way turnover rather than 2.24% on both entry/exit trades).
- **Realistic Net Performance:** After correcting the clerical bad tick and charging the full CSE statutory 2.24% round-trip trading fees, LightGBM 20d delivers **55.55% net CAGR** (Sharpe 1.21, Max Drawdown -20.34%), outperforming the ASPI benchmark (46.67%).
- **Execution Realism:** Testing next-day (T+1 Open) execution with realistic bid-ask spread assumptions confirms LightGBM remains viable and profitable (**52.07% to 58.53% net CAGR**) as long as total round-trip transaction friction is $\le 3.24\%$.

## Engineering principles

- **Point-in-time correctness:** information must have been publicly available by the recommendation timestamp.
- **Chronological validation:** use walk-forward validation and purging/embargo for overlapping forward-return targets.
- **Survivorship awareness:** preserve delisted securities where source data permits.
- **Corporate-action correctness:** account for splits, rights issues, dividends and adjusted prices.
- **Realistic trading:** model turnover, fees, liquidity, spread, slippage and execution constraints.
- **Explainability:** communicate important drivers, uncertainty and risk alongside rankings.
- **Horizon-specific models:** do not force one model or feature set to serve all investment strategies.

## Repository structure

```text
Stock_Pulse/
├── 01_CSE_Exploratory_Data_Analysis.ipynb
├── 02_Feature_Engineering_and_Targets.ipynb
├── 03_Recommendation_System.ipynb
├── app.py
├── utils/
├── scripts/
├── docs/
│   ├── data_enrichment/
│   ├── research/
│   └── model_evaluation/
├── assets/
├── tests/
├── requirements.txt
└── README.md
```

This is an intended organization guide; not every directory or artifact is necessarily present on every branch.

## Getting started

1. Use Python 3.10 or newer.
2. Clone the repository:

   ```bash
   git clone https://github.com/iamdushanl/Stock_Pulse.git
   cd Stock_Pulse
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\\Scripts\\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Configure local data paths using the documentation under `docs/`. Raw CSE datasets and generated Parquet/model artifacts are not stored in this repository.

## Data and artifact policy

Do not commit:
- Raw CSE datasets or downloaded financial-report PDFs unless redistribution rights are confirmed.
- Credentials, API keys, tokens, or `.env` files.
- Large generated CSV/Parquet files, model binaries, or local caches.
- Broker/account or other sensitive information.

Commit reproducible source code, tests, configuration examples, methodology, data dictionaries, lightweight aggregated reports, and non-sensitive metadata. Use GitHub Releases or controlled external storage for large artifacts only where licensing and access rights permit.

## Research and model governance

Before a model is approved:
1. Reproduce its backtest independently from saved predictions, trades and source data.
2. Verify price adjustment, dividends, corporate actions and historical universe membership.
3. Recalculate costs from actual portfolio weight changes and test spread/slippage.
4. Compare with ASPI and simple baselines.
5. Evaluate performance by year and market regime.
6. Preserve an untouched final test period.
7. Record dataset versions, feature schema, code commit, random seed and model parameters.

## Roadmap

- [x] Complete initial EDA
- [x] Build CSE data enrichment and financial extraction pipeline
- [x] Generate three horizon-specific feature matrices
- [x] Audit point-in-time and feature integrity
- [x] Benchmark initial short-term model candidates
- [x] Independently verify Phase 4.1 portfolio backtests
- [ ] Expand historical point-in-time fundamentals
- [ ] Build and validate medium-term growth engine
- [ ] Build and validate long-term total-return engine
- [ ] Expand from SP20 to the eligible full CSE universe
- [ ] Integrate approved models into the recommendation application
- [ ] Add monitoring, retraining and model-drift checks

## License and data rights

See [LICENSE](LICENSE) for the code license. The code license does not grant rights to redistribute CSE datasets, exchange disclosures, or third-party financial documents. Users are responsible for observing source terms and applicable law.

---

**Academic group project – Colombo Stock Exchange quantitative research**
