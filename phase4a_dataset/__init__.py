"""
Stockpulse Phase 4A - Short-Term Modeling Dataset Preparation & Target Engineering
==================================================================================

Converts the validated Phase 3D short-term integrated research universe into a
leakage-controlled, reproducible, chronologically split modeling dataset ready for
Phase 5A short-term predictive modeling.
"""

from phase4a_dataset.calendar_and_targets import (
    HORIZONS,
    MAX_HORIZON,
    EMBARGO_SESSIONS,
    PURGE_SESSIONS,
    TOTAL_GAP_SESSIONS,
    MIN_CROSS_SECTION_SIZE,
    build_trading_calendar_and_benchmarks,
    compute_forward_targets_and_labels,
    load_astri_benchmark,
)
from phase4a_dataset.features_eligibility_and_splits import (
    DEFAULT_CV_FOLD_SPECS,
    FINAL_TEST_START_DATE,
    PENNY_CLOSE_THRESHOLD_LKR,
    PHASE4A_OUTPUT_FILES,
    assign_chronological_purged_splits,
    build_derived_scale_free_features,
    build_eligibility_and_exclusion_audit,
    build_short_term_feature_registry,
    evaluate_sensitivity_and_target_quality,
    generate_phase4a_visualizations,
    run_phase4a_leakage_and_perturbation_tests,
    run_phase4a_pipeline,
)

__all__ = [
    "HORIZONS",
    "MAX_HORIZON",
    "PURGE_SESSIONS",
    "EMBARGO_SESSIONS",
    "TOTAL_GAP_SESSIONS",
    "MIN_CROSS_SECTION_SIZE",
    "DEFAULT_CV_FOLD_SPECS",
    "FINAL_TEST_START_DATE",
    "PENNY_CLOSE_THRESHOLD_LKR",
    "PHASE4A_OUTPUT_FILES",
    "load_astri_benchmark",
    "build_trading_calendar_and_benchmarks",
    "compute_forward_targets_and_labels",
    "build_derived_scale_free_features",
    "build_short_term_feature_registry",
    "build_eligibility_and_exclusion_audit",
    "assign_chronological_purged_splits",
    "evaluate_sensitivity_and_target_quality",
    "run_phase4a_leakage_and_perturbation_tests",
    "generate_phase4a_visualizations",
    "run_phase4a_pipeline",
]
