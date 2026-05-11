"""
Task 1 — Data Preparation
=========================
Loads the raw dataset, fixes formats, creates was_missing flags,
engineers 3 domain-motivated features, and exports two raw pickles:

    - dataset_lr_raw.pkl  : for Logistic Regression pipeline
                            (no encoding, no imputation — done in train_lr)
    - dataset_lgbm.pkl    : for LightGBM
                            (categoricals as dtype "category", missing native)

No imputation or encoding is done here to avoid any risk of data leakage.
All statistically-sensitive operations (imputation, encoding, scaling,
multicollinearity drop) are performed AFTER the train/test split in the
respective training tasks.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "churn"

# Robust drop of identifier column (may appear as Customer_ID or vpn_key*)
ID_PATTERNS = ["Customer_ID", "vpn_key"]

# Categorical columns (string dtype — not binary 0/1)
CATEGORICAL_COLS = [
    "new_cell", "crclscod", "asl_flag", "prizm_social_one",
    "area", "dualband", "refurb_new", "hnd_webcap",
    "ownrent", "dwlltype", "marital", "infobase",
    "HHstatin", "dwllsize", "ethnic", "creditcd",
    "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17",
]

# Binary categorical columns (0/1 but semantically categorical)
BINARY_CAT_COLS = ["truck", "rv", "forgntvl"]

# Thresholds for missing value treatment
MISSING_DROP_THRESHOLD = 0.50    # drop column if >50% missing
MISSING_FLAG_THRESHOLD = 0.20    # add was_missing flag if >20% missing

# Feature engineering threshold
OLD_PHONE_THRESHOLD = 500        # eqpdays > 500 → old phone flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fix_decimal_separator(series: pd.Series) -> pd.Series:
    """
    Converts comma decimal separator to dot and casts to numeric.
    e.g. '23,99' -> 23.99
    Only applied to object columns that are not categorical.
    """
    if series.dtype == object:
        n_non_null = int(series.notna().sum())
        series = series.astype(str).str.replace(",", ".", regex=False)
        series = pd.to_numeric(series, errors="coerce")
        if n_non_null > 0:
            n_new_nan = int(series.isna().sum()) - (len(series) - n_non_null)
            if n_new_nan > 0 and n_new_nan / n_non_null > 0.05:
                logger.warning(
                    f"fix_decimal_separator: column '{series.name}' — "
                    f"{n_new_nan} new NaNs ({n_new_nan/n_non_null*100:.1f}% of non-null) — "
                    "column may contain mixed non-numeric values."
                )
    return series


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly drops identifier columns matching any pattern in ID_PATTERNS.
    Handles both 'Customer_ID' and 'vpn_keyCustomer_ID' variants.
    """
    cols_to_drop = [
        c for c in df.columns
        if any(pattern in c for pattern in ID_PATTERNS)
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped identifier columns: {cols_to_drop}")
    return df


def create_was_missing_flags(df: pd.DataFrame,
                              missing_pct: pd.Series) -> pd.DataFrame:
    """
    Creates binary was_missing flag for columns with missing rate
    between MISSING_FLAG_THRESHOLD and MISSING_DROP_THRESHOLD.

    These flags capture whether missingness itself is informative
    (e.g. customers who don't provide demographic data may differ
    systematically from those who do).

    Args:
        df:          dataframe
        missing_pct: series with missing percentage per column

    Returns:
        dataframe with additional *_was_missing columns
    """
    flag_cols = missing_pct[
        (missing_pct > MISSING_FLAG_THRESHOLD) &
        (missing_pct <= MISSING_DROP_THRESHOLD)
    ].index.tolist()

    for col in flag_cols:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isnull().astype(int)

    logger.info(f"Created was_missing flags for {len(flag_cols)} columns: {flag_cols}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 3 domain-motivated features grounded in EDA findings.

    Feature 1 — recent_usage_delta:
        Difference between 3-month and 6-month average minutes of use.
        Captures the discrete derivative of usage behaviour over time.
        A declining trend signals potential churn intent.
        Note: avg6mou likely includes the last 3 months, so this is an
        approximation of the trend rather than a clean "recent vs past"
        comparison. Collinearity with change_mou will be checked in train_lr.

    Feature 2 — drop_rate:
        Ratio of dropped voice calls to placed voice calls.
        Proxy for perceived service quality — frustrated customers churn.
        Pure domain knowledge: cannot be derived from correlation alone.

    Feature 3 — old_phone:
        Binary flag: equipment age > 500 days.
        Motivated by EDA: eqpdays is the most correlated numeric feature
        with churn (Pearson r = 0.11). Customers with old phones want
        to switch operator for a better device deal.
        Non-linearised via threshold to capture the step-change effect.
    """
    logger.info("Engineering domain-motivated features...")

    # Feature 1
    if "avg3mou" in df.columns and "avg6mou" in df.columns:
        df["recent_usage_delta"] = df["avg3mou"] - df["avg6mou"]
        logger.info("  ① recent_usage_delta = avg3mou - avg6mou")
    else:
        logger.warning("  ① recent_usage_delta skipped — avg3mou or avg6mou missing")

    # Feature 2
    if "drop_vce_Mean" in df.columns and "plcd_vce_Mean" in df.columns:
        df["drop_rate"] = (
            pd.to_numeric(df["drop_vce_Mean"], errors="coerce") /
            (pd.to_numeric(df["plcd_vce_Mean"], errors="coerce") + 1)
        )
        logger.info("  ② drop_rate = drop_vce_Mean / (plcd_vce_Mean + 1)")
    else:
        logger.warning("  ② drop_rate skipped — required columns missing")

    # Feature 3
    if "eqpdays" in df.columns:
        df["old_phone"] = np.where(
            df["eqpdays"].isna(),
            np.nan,
            (df["eqpdays"] > OLD_PHONE_THRESHOLD).astype(float),
        )
        logger.info(f"  ③ old_phone = (eqpdays > {OLD_PHONE_THRESHOLD}) — NaN preserved")
    else:
        logger.warning("  ③ old_phone skipped — eqpdays missing")

    return df


def apply_domain_validation_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies lightweight domain-aware validation rules.

    These are not generic statistical outlier removals. They only correct
    semantically inconsistent or clearly unrealistic edge cases identified
    during dataset inspection.

    Rules:
    - eqpdays cannot be negative -> set to NaN
    - uniqsubs extreme values above 13 -> clip to 13
    - actvsubs extreme values above 11 -> clip to 11
    - drop_vce_Mean cannot exceed plcd_vce_Mean -> cap drop_vce_Mean at plcd_vce_Mean
    """
    logger.info("Applying domain-aware validation rules...")

    if "eqpdays" in df.columns:
        n_invalid = int((df["eqpdays"] < 0).sum())

        if n_invalid > 0:
            df.loc[df["eqpdays"] < 0, "eqpdays"] = np.nan

            logger.info(
                f"  Fixed {n_invalid} negative eqpdays values -> NaN"
            )

    if "uniqsubs" in df.columns:
        n_clipped = int((df["uniqsubs"] > 13).sum())

        if n_clipped > 0:
            df["uniqsubs"] = df["uniqsubs"].clip(upper=13)

            logger.info(
                f"  Clipped {n_clipped} uniqsubs values above 13"
            )

    if "actvsubs" in df.columns:
        n_clipped = int((df["actvsubs"] > 11).sum())

        if n_clipped > 0:
            df["actvsubs"] = df["actvsubs"].clip(upper=11)

            logger.info(
                f"  Clipped {n_clipped} actvsubs values above 11"
            )

    if {"drop_vce_Mean", "plcd_vce_Mean"}.issubset(df.columns):

        mask = df["drop_vce_Mean"] > df["plcd_vce_Mean"]

        n_fixed = int(mask.sum())

        if n_fixed > 0:
            df.loc[mask, "drop_vce_Mean"] = (
                df.loc[mask, "plcd_vce_Mean"]
            )

            logger.info(
                f"  Fixed {n_fixed} rows where dropped calls exceeded placed calls"
            )

    return df


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def load_and_prepare(data_path: str,
                     output_lr_path: str,
                     output_lgbm_path: str) -> None:
    """
    Full data preparation pipeline. Produces two dataset pickles
    optimised for Logistic Regression and LightGBM respectively.

    Args:
        data_path:        path to raw dataset.csv
        output_lr_path:   path for dataset_lr_raw.pkl
        output_lgbm_path: path for dataset_lgbm.pkl
    """
    logger.info("=" * 60)
    logger.info("TASK 1: Data Preparation")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, sep=";", low_memory=False)
    logger.info(f"Raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 1b. Robustness checks
    # ------------------------------------------------------------------
    if df.empty:
        raise ValueError("Dataset is empty — aborting preparation.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset. "
                         f"Available columns: {list(df.columns)}")

    _target_raw = df[TARGET_COL]
    if _target_raw.isnull().any():
        n_null = int(_target_raw.isnull().sum())
        raise ValueError(f"Target column '{TARGET_COL}' contains {n_null} NaN values — aborting.")

    _n_classes = _target_raw.nunique()
    if _n_classes < 2:
        raise ValueError(f"Target column '{TARGET_COL}' has only {_n_classes} unique class(es) — "
                         "need at least 2 for classification.")

    _dup_count = int(df.duplicated().sum())
    logger.info(f"Duplicate rows: {_dup_count:,} ({_dup_count / len(df) * 100:.2f}% of dataset)")

    # ------------------------------------------------------------------
    # 2. Fix decimal separators on non-categorical numeric columns
    # ------------------------------------------------------------------
    logger.info("Fixing decimal separators...")
    all_cat = CATEGORICAL_COLS + BINARY_CAT_COLS + [TARGET_COL]
    for col in df.columns:
        if col not in all_cat:
            df[col] = fix_decimal_separator(df[col])

    # ------------------------------------------------------------------
    # 2b. Domain-aware validation rules
    # ------------------------------------------------------------------
    df = apply_domain_validation_rules(df)

    # ------------------------------------------------------------------
    # 3. Drop identifier columns
    # ------------------------------------------------------------------
    _dropped_id_cols = [c for c in df.columns if any(p in c for p in ID_PATTERNS)]
    df = drop_id_columns(df)

    # Warn about any remaining object columns not declared as categorical
    _known_cats = set(CATEGORICAL_COLS + BINARY_CAT_COLS + [TARGET_COL])
    _undeclared_obj = [c for c in df.select_dtypes(include="object").columns if c not in _known_cats]
    if _undeclared_obj:
        logger.warning(
            f"Object dtype columns not in CATEGORICAL_COLS or BINARY_CAT_COLS: "
            f"{_undeclared_obj} — possible forgotten categorical columns."
        )

    # ------------------------------------------------------------------
    # 4. Separate target
    # ------------------------------------------------------------------
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    churn_rate = y.mean()
    logger.info(
        f"Target: churn=1: {y.sum():,} ({churn_rate*100:.1f}%), "
        f"churn=0: {(1-y).sum():,} ({(1-churn_rate)*100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 5. Missing value analysis
    # ------------------------------------------------------------------
    missing_pct = (X.isnull().sum() / len(X))
    missing_report = (missing_pct * 100).round(1)
    missing_report = missing_report[missing_report > 0].sort_values(ascending=False)
    logger.info(f"Missing value report:\n{missing_report.to_string()}")

    # ------------------------------------------------------------------
    # 6. Drop columns with >50% missing
    # ------------------------------------------------------------------
    high_missing = missing_pct[missing_pct > MISSING_DROP_THRESHOLD].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)
        logger.info(f"Dropped columns (>{MISSING_DROP_THRESHOLD*100:.0f}% missing): {high_missing}")
    else:
        logger.info("No columns dropped for excessive missing values")

    # Recompute after drop
    missing_pct = X.isnull().sum() / len(X)

    # ------------------------------------------------------------------
    # 7. Create was_missing flags (20-50% missing)
    # ------------------------------------------------------------------
    X = create_was_missing_flags(X, missing_pct)

    # ------------------------------------------------------------------
    # 8. Feature engineering
    # ------------------------------------------------------------------
    X = engineer_features(X)

    # ------------------------------------------------------------------
    # 9. Log final feature count
    # ------------------------------------------------------------------
    logger.info(f"Features after preparation: {X.shape[1]}")
    logger.info(f"  Categorical: {len([c for c in CATEGORICAL_COLS if c in X.columns])}")
    logger.info(f"  Binary cat:  {len([c for c in BINARY_CAT_COLS if c in X.columns])}")
    logger.info(f"  Numeric:     {X.select_dtypes(include='number').shape[1]}")

    # ------------------------------------------------------------------
    # 10. Build dataset_lr_raw.pkl
    #     Raw — no imputation, no encoding, no scaling
    #     All operations done AFTER split in train_lr task
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_lr_path), exist_ok=True)

    payload_lr = {
        "X": X.copy(),
        "y": y,
        "categorical_cols": [c for c in CATEGORICAL_COLS if c in X.columns],
        "binary_cat_cols": [c for c in BINARY_CAT_COLS if c in X.columns],
        "feature_names": list(X.columns),
    }
    with open(output_lr_path, "wb") as f:
        pickle.dump(payload_lr, f)
    logger.info(f"Saved dataset_lr_raw.pkl to: {output_lr_path}")

    # ------------------------------------------------------------------
    # 11. Build dataset_lgbm.pkl
    #     Categoricals as dtype "category" — LightGBM handles natively
    #     Missing values kept as NaN — LightGBM handles natively
    #     was_missing flags kept for additional signal
    # ------------------------------------------------------------------
    X_lgbm = X.copy()

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_lgbm.columns]
    bin_cat_present = [c for c in BINARY_CAT_COLS if c in X_lgbm.columns]

    for col in cat_cols_present:
        # Preserve NaN as actual NaN — do NOT cast to str (would convert NaN to "NAN")
        # LightGBM handles NaN in categorical columns natively
        X_lgbm[col] = X_lgbm[col].where(X_lgbm[col].isna(),
                                          X_lgbm[col].astype(str).str.strip().str.upper())
        X_lgbm[col] = X_lgbm[col].astype("category")

    for col in bin_cat_present:
        # Same: preserve NaN for native LightGBM handling
        X_lgbm[col] = X_lgbm[col].astype("category")

    logger.info(
        f"LightGBM dataset: {len(cat_cols_present + bin_cat_present)} "
        f"categorical features set as dtype 'category'"
    )

    os.makedirs(os.path.dirname(output_lgbm_path), exist_ok=True)

    payload_lgbm = {
        "X": X_lgbm,
        "y": y,
        "categorical_cols": cat_cols_present + bin_cat_present,
        "feature_names": list(X_lgbm.columns),
    }
    with open(output_lgbm_path, "wb") as f:
        pickle.dump(payload_lgbm, f)
    logger.info(f"Saved dataset_lgbm.pkl to: {output_lgbm_path}")

    # ------------------------------------------------------------------
    # 12. Preparation report JSON
    # ------------------------------------------------------------------
    _missing_final = (X.isnull().sum() / len(X) * 100).round(2)
    _missing_summary = {
        col: float(val)
        for col, val in _missing_final[_missing_final > 0]
                           .sort_values(ascending=False).items()
    }
    _was_missing_flags = [c for c in X.columns if c.endswith("_was_missing")]
    _engineered = [c for c in ["recent_usage_delta", "drop_rate", "old_phone"]
                   if c in X.columns]

    preparation_report = {
        "dataset_shape": {"rows": int(X.shape[0]), "columns": int(X.shape[1])},
        "churn_rate": round(float(y.mean()), 4),
        "churn_count": int(y.sum()),
        "no_churn_count": int((1 - y).sum()),
        "duplicate_row_count": _dup_count,
        "dropped_columns": high_missing,
        "dropped_identifier_columns": _dropped_id_cols,
        "dropped_high_missing_columns": high_missing,
        "was_missing_flags": _was_missing_flags,
        "engineered_features": _engineered,
        "missing_value_summary": _missing_summary,
        "final_feature_count": int(X.shape[1]),
        "categorical_feature_count": len([c for c in CATEGORICAL_COLS if c in X.columns]),
        "binary_cat_feature_count": len([c for c in BINARY_CAT_COLS if c in X.columns]),
        "numeric_feature_count": int(X.select_dtypes(include="number").shape[1]),
    }

    _models_dir = os.path.dirname(output_lr_path)
    _airflow_home = os.path.dirname(_models_dir)
    _report_path = os.path.join(_airflow_home, "outputs", "preparation_report.json")
    os.makedirs(os.path.dirname(_report_path), exist_ok=True)
    with open(_report_path, "w") as f:
        json.dump(preparation_report, f, indent=2)
    logger.info(f"Preparation report saved to: {_report_path}")

    logger.info("=" * 60)
    logger.info("TASK 1 completed successfully")
    logger.info("=" * 60)
