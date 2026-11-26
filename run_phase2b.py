"""
Stockpulse Phase 2B Entrypoint – Feature Quality, Market Context & Readiness Audit
=================================================================================
Reproducibly loads Phase 1B and Phase 2A datasets through `utils.data_loader`,
validates required inputs, executes Workstreams A–I, generates 11 Phase 2B
deliverables in `Dataset/processed/phase2b/` plus visualizations in
`reports/phase2/visualizations/`, and records full manifest & SHA-256 checksums.

Does NOT modify Phase 1A, Phase 1B, or Phase 2A outputs, train ML models,
or produce stock recommendations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2b_audit import (
    assess_strategy_readiness,
    audit_coverage_and_missingness,
    audit_dataset_integrity,
    audit_feature_redundancy,
    audit_liquidity_and_tradability,
    audit_survivorship_and_universe,
    generate_visualizations,
    validate_corporate_actions,
    validate_market_relative,
)
from utils.data_loader import (
    DEFAULT_PHASE1B_PATH,
    DEFAULT_PHASE1_PATH,
    DEFAULT_PHASE2_PATH,
    DEFAULT_RAW_SOURCE_PATH,
    load_cse_corporate_actions,
    load_cse_daily_prices,
    load_cse_market_calendar,
    load_cse_sector_mapping,
    load_cse_security_master,
    load_market_indices,
    load_phase2a_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase2b")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_required_inputs(
    p1_dir: Path, p1b_dir: Path, p2_dir: Path, raw_dir: Path
) -> dict[str, Path]:
    required = {
        "phase1_calendar": p1_dir / "cse_market_calendar.parquet",
        "phase1b_integrated": p1b_dir / "cse_daily_prices_integrated.parquet",
        "phase1b_split_adjusted": p1b_dir / "cse_daily_prices_split_adjusted.parquet",
        "phase1b_security_master": p1b_dir / "cse_security_master_enriched.parquet",
        "phase1b_ca_validated": p1b_dir / "cse_corporate_actions_validated.parquet",
        "phase1b_ca_factors": p1b_dir / "corporate_action_adjustment_factors.parquet",
        "phase1b_sector_mapping": p1b_dir / "cse_sector_mapping.parquet",
        "phase2a_technical": p2_dir / "cse_technical_features.parquet",
        "phase2a_price_volume": p2_dir / "cse_price_volume_features.parquet",
        "phase2a_market_relative": p2_dir / "cse_market_relative_features.parquet",
        "phase2a_metadata": p2_dir / "technical_indicator_metadata.csv",
        "raw_market_indices": raw_dir / "07Market Indices - Daily.xls",
    }
    missing = [f"{k}: {v}" for k, v in required.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Phase 1B/2A input file(s) missing:\n  " + "\n  ".join(missing)
        )
    return required


def run_phase2b() -> dict[str, Any]:
    t0 = time.time()
    p1_dir = Path(DEFAULT_PHASE1_PATH)
    p1b_dir = Path(DEFAULT_PHASE1B_PATH)
    p2_dir = Path(DEFAULT_PHASE2_PATH)
    raw_dir = Path(DEFAULT_RAW_SOURCE_PATH)

    out_dir = REPO_ROOT / "Dataset" / "processed" / "phase2b"
    viz_dir = REPO_ROOT / "reports" / "phase2" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("Stockpulse Phase 2B – Full CSE Feature Quality & Training Readiness Audit")
    logger.info("=" * 72)

    req_files = _validate_required_inputs(p1_dir, p1b_dir, p2_dir, raw_dir)

    # 1. Load datasets via centralized `utils.data_loader` functions
    logger.info("[1/7] Loading Phase 1B and Phase 2A datasets via utils.data_loader...")
    df_adj = load_cse_daily_prices(path=req_files["phase1b_split_adjusted"])
    df_raw = load_cse_daily_prices(path=req_files["phase1b_integrated"])
    sec_master = load_cse_security_master(path=req_files["phase1b_security_master"])
    ca_validated = load_cse_corporate_actions(path=req_files["phase1b_ca_validated"])
    factors_audit = pd.read_parquet(req_files["phase1b_ca_factors"])
    sector_map = load_cse_sector_mapping(path=req_files["phase1b_sector_mapping"])
    market_cal = load_cse_market_calendar(path=req_files["phase1_calendar"])
    market_indices = load_market_indices(str(raw_dir))

    tech_df = load_phase2a_features("technical", path=p2_dir)
    pv_df = load_phase2a_features("price_volume", path=p2_dir)
    mr_df = load_phase2a_features("market_relative", path=p2_dir)
    meta_df = pd.read_csv(req_files["phase2a_metadata"])

    # 2. Workstream A – Dataset Integrity Audit
    logger.info("[2/7] Executing Workstream A – Dataset Integrity Audit...")
    integrity_report = audit_dataset_integrity(
        df_adj=df_adj,
        df_raw=df_raw,
        sec_master=sec_master,
        tech_df=tech_df,
        pv_df=pv_df,
        mr_df=mr_df,
        meta_df=meta_df,
    )
    with open(out_dir / "phase2b_integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2, default=str)

    # 3. Workstream B – Coverage & Missingness Taxonomy
    logger.info("[3/7] Executing Workstream B – Coverage & Missingness Analysis...")
    coverage_df, missingness_df, annual_cov_df = audit_coverage_and_missingness(
        df_adj=df_adj,
        sec_master=sec_master,
        sector_map=sector_map,
        market_indices=market_indices,
        tech_df=tech_df,
        pv_df=pv_df,
        mr_df=mr_df,
        meta_df=meta_df,
    )
    coverage_df.to_csv(out_dir / "feature_coverage_report.csv", index=False)
    missingness_df.to_csv(out_dir / "feature_missingness_report.csv", index=False)

    # 4. Workstream C – Liquidity & Tradability Analysis
    logger.info("[4/7] Executing Workstream C – Liquidity & Tradability Analysis...")
    liq_df, liq_summary = audit_liquidity_and_tradability(
        df_adj=df_adj,
        sec_master=sec_master,
        market_calendar=market_cal,
        pv_df=pv_df,
    )
    liq_df.to_csv(out_dir / "liquidity_analysis.csv", index=False)

    # 5. Workstreams D & E – Market-Relative & Corporate Action Audits
    logger.info("[5/7] Executing Workstreams D & E – Market-Relative & Corporate Action Validation...")
    mr_val_df, mr_summary = validate_market_relative(
        df_adj=df_adj,
        mr_df=mr_df,
        market_indices=market_indices,
        sector_map=sector_map,
        raw_dir=raw_dir,
    )
    mr_val_df.to_csv(out_dir / "market_relative_validation.csv", index=False)

    ca_val_df, ca_summary = validate_corporate_actions(
        df_adj=df_adj,
        df_raw=df_raw,
        ca_validated=ca_validated,
        factors_audit=factors_audit,
        pv_df=pv_df,
    )
    ca_val_df.to_csv(out_dir / "corporate_action_validation.csv", index=False)

    # 6. Workstreams F & G – Feature Redundancy & Survivorship Bias Audit
    logger.info("[6/7] Executing Workstreams F & G – Feature Redundancy & Survivorship Audit...")
    corr_df, red_df, red_summary = audit_feature_redundancy(
        tech_df=tech_df,
        pv_df=pv_df,
        mr_df=mr_df,
        meta_df=meta_df,
        liq_df=liq_df,
    )
    corr_df.to_csv(out_dir / "feature_correlation_matrix.csv", index=False)
    red_df.to_csv(out_dir / "feature_redundancy_report.csv", index=False)

    surv_df, surv_summary = audit_survivorship_and_universe(
        df_adj=df_adj,
        sec_master=sec_master,
        ca_validated=ca_validated,
        liq_df=liq_df,
    )
    surv_df.to_csv(out_dir / "security_history_coverage.csv", index=False)

    # 7. Workstreams H & I – Strategy Readiness & Visualizations
    logger.info("[7/7] Executing Workstreams H & I – Strategy Readiness Assessment & Visualizations...")
    readiness = assess_strategy_readiness(
        repo_root=REPO_ROOT,
        integrity_report=integrity_report,
        coverage_df=coverage_df,
        liq_summary=liq_summary,
        mr_summary=mr_summary,
        ca_summary=ca_summary,
        red_summary=red_summary,
        surv_summary=surv_summary,
        ca_validated=ca_validated,
    )
    readiness["liquidity_summary"] = liq_summary
    readiness["market_relative_summary"] = mr_summary
    readiness["corporate_action_summary"] = ca_summary
    readiness["redundancy_summary"] = red_summary
    readiness["survivorship_summary"] = surv_summary
    with open(out_dir / "strategy_readiness_assessment.json", "w", encoding="utf-8") as f:
        json.dump(readiness, f, indent=2, default=str)

    viz_files = generate_visualizations(
        viz_dir=viz_dir,
        coverage_df=coverage_df,
        annual_cov_df=annual_cov_df,
        liq_df=liq_df,
        liq_summary=liq_summary,
        corr_df=corr_df,
        surv_summary=surv_summary,
        readiness=readiness,
    )

    output_names = [
        "phase2b_integrity_report.json",
        "feature_coverage_report.csv",
        "feature_missingness_report.csv",
        "liquidity_analysis.csv",
        "market_relative_validation.csv",
        "corporate_action_validation.csv",
        "feature_correlation_matrix.csv",
        "feature_redundancy_report.csv",
        "security_history_coverage.csv",
        "strategy_readiness_assessment.json",
    ]
    checksums = {name: _sha256_file(out_dir / name) for name in output_names}

    manifest = {
        "phase": "2B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "input_paths": {k: str(v) for k, v in req_files.items()},
        "row_count": int(len(df_adj)),
        "unique_security_ids": int(df_adj["SecurityID"].nunique()),
        "unique_company_codes": int(df_adj["CompanyCode"].nunique()),
        "date_range": [
            str(pd.to_datetime(df_adj["Date"]).min().date()),
            str(pd.to_datetime(df_adj["Date"]).max().date()),
        ],
        "configuration": {
            "liquidity_eval_basis": "trailing_20_60_120_observations_with_calendar_frequency",
            "redundancy_sample_regime": "2017_2025_MainType_N_uniform_stride",
            "beta_official_comparison_sheet": "2025Q4",
        },
        "integrity_status": integrity_report["cross_table_consistency"]["overall_integrity_status"],
        "strategy_verdicts": {
            "short_term_trading_0_3m": readiness["strategies"]["short_term_trading_0_3m"]["status"],
            "medium_term_growth_3_12m": readiness["strategies"]["medium_term_growth_3_12m"]["status"],
            "long_term_wealth_1_5y": readiness["strategies"]["long_term_wealth_1_5y"]["status"],
        },
        "output_checksums_sha256": checksums,
        "visualizations": viz_files,
        "output_files": output_names + ["phase2b_manifest.json"],
    }
    with open(out_dir / "phase2b_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(
        "Phase 2B COMPLETE in %.2fs | Integrity=%s | Verdicts: ST=%s, MT=%s, LT=%s",
        manifest["elapsed_seconds"],
        manifest["integrity_status"],
        manifest["strategy_verdicts"]["short_term_trading_0_3m"],
        manifest["strategy_verdicts"]["medium_term_growth_3_12m"],
        manifest["strategy_verdicts"]["long_term_wealth_1_5y"],
    )
    return manifest


if __name__ == "__main__":
    run_phase2b()
