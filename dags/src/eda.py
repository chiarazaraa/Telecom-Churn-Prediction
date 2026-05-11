"""
EDA Pipeline — Exploratory Data Analysis
=========================================
Standalone analysis script — run once before the main churn pipeline.
Produces plots and statistics saved to outputs/eda/ for use in
the presentation and technical report.

Analyses performed:
    1. Target distribution
    2. Missing values analysis (bar plot)
    3. Numeric features distributions + churn overlay
    4. Outlier detection (Z-score)
    5. Churn rate by categorical variables
    6. Correlation heatmap (numeric features)
    7. Key feature distributions split by churn
    8. Summary statistics saved as JSON
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# Colour palette
BLUE   = "#2563eb"
CORAL  = "#dc2626"
GRAY   = "#6b7280"
GREEN  = "#059669"
PURPLE = "#7c3aed"
AMBER  = '#d97706'

# Same categorical columns as preparation.py
CATEGORICAL_COLS = [
    "new_cell", "crclscod", "asl_flag", "prizm_social_one",
    "area", "dualband", "refurb_new", "hnd_webcap",
    "ownrent", "dwlltype", "marital", "infobase",
    "HHstatin", "dwllsize", "ethnic", "creditcd",
    "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17",
]
BINARY_CAT_COLS = ["truck", "rv"]
TARGET_COL      = "churn"
ID_PATTERNS     = ["Customer_ID", "vpn_key"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fix_decimal_separator(series: pd.Series) -> pd.Series:
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
                    "may contain mixed non-numeric values."
                )
    return series


def safe_savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_target_distribution(y: pd.Series, output_dir: str) -> None:
    counts    = y.value_counts().reindex([0, 1], fill_value=0)
    labels    = ["No Churn", "Churn"]
    colors    = [BLUE, CORAL]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, alpha=0.85, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}\n({v/len(y)*100:.1f}%)",
                     ha="center", fontsize=11)
    axes[0].set_title("Churn Distribution — Count")
    axes[0].set_ylabel("Number of customers")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Churn Distribution — Proportion")

    fig.suptitle("Target Variable Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "01_target_distribution.png"))


def plot_missing_values(df: pd.DataFrame, output_dir: str) -> None:
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        logger.info("No missing values found — skipping missing values plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.35)))
    colors = [CORAL if v > 50 else AMBER if v > 20 else BLUE
              for v in missing_pct.values]

    bars = ax.barh(missing_pct.index, missing_pct.values,
                   color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(50, color=CORAL,  lw=1.5, linestyle="--", label="> 50% → drop")
    ax.axvline(20, color=PURPLE, lw=1.5, linestyle="--", label="> 20% → was_missing flag")

    for bar, val in zip(bars, missing_pct.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Missing percentage (%)")
    ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "02_missing_values.png"))


def plot_numeric_distributions(df: pd.DataFrame, num_cols: list,
                                output_dir: str) -> None:
    """Histograms of top numeric features split by churn."""
    # Select top 12 most correlated with churn
    corr = df[num_cols].corrwith(df[TARGET_COL].astype(float)).abs()
    top_cols = corr.sort_values(ascending=False).head(12).index.tolist()

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        ax = axes[i]
        no_churn = df[df[TARGET_COL] == 0][col].dropna()
        churned  = df[df[TARGET_COL] == 1][col].dropna()

        ax.hist(no_churn, bins=30, alpha=0.6, color=BLUE,
                label="No Churn", density=True)
        ax.hist(churned,  bins=30, alpha=0.6, color=CORAL,
                label="Churn",    density=True)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(top_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top 12 Numeric Features — Distribution by Churn",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "03_numeric_distributions.png"))


def plot_outlier_detection(df: pd.DataFrame, num_cols: list,
                            output_dir: str) -> None:
    """Z-score based outlier detection for numeric features."""
    outlier_pct = {}
    for col in num_cols:
        clean = df[col].dropna()
        if len(clean) < 10:
            continue
        z = np.abs(stats.zscore(clean))
        outlier_pct[col] = float((z > 3).sum() / len(clean) * 100)

    outlier_series = pd.Series(outlier_pct).sort_values(ascending=False)
    outlier_series = outlier_series[outlier_series > 1.0]  # show only >1%

    if outlier_series.empty:
        logger.info("No significant outliers found (>1%) — skipping outlier plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(outlier_series) * 0.4)))
    ax.barh(outlier_series.index, outlier_series.values,
            color=CORAL, alpha=0.8, edgecolor="white")
    ax.axvline(5, color=GRAY, lw=1.5, linestyle="--", label="5% threshold")

    for i, (idx, val) in enumerate(outlier_series.items()):
        ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Percentage of outliers (|Z-score| > 3)")
    ax.set_title("Outlier Detection by Feature (Z-score method)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "04_outlier_detection.png"))


def plot_churn_by_categorical(df: pd.DataFrame, output_dir: str) -> None:
    """Churn rate for each categorical variable."""
    cat_cols_present = [c for c in CATEGORICAL_COLS + BINARY_CAT_COLS
                        if c in df.columns]

    # Select top 9 most discriminative (highest variance in churn rate)
    discriminative = []
    for col in cat_cols_present:
        try:
            rates = df.groupby(col)[TARGET_COL].mean()
            if len(rates) >= 2:
                discriminative.append((col, rates.std()))
        except Exception:
            pass

    discriminative.sort(key=lambda x: x[1], reverse=True)
    top_cats = [c for c, _ in discriminative[:9]]

    if not top_cats:
        logger.warning("No categorical features for churn rate plot")
        return

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    global_rate = df[TARGET_COL].mean()

    for i, col in enumerate(top_cats):
        ax = axes[i]
        tmp = df[[col, TARGET_COL]].copy()
        tmp[col] = tmp[col].fillna("MISSING")
        rates = (tmp.groupby(col)[TARGET_COL]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "churn_rate", "count": "n"}))
        rates = rates[rates["n"] > 50].sort_values("churn_rate", ascending=True)

        colors = [CORAL if r > global_rate else BLUE
                  for r in rates["churn_rate"]]
        ax.barh(rates.index.astype(str), rates["churn_rate"],
                color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(global_rate, color=GRAY, lw=1.5, linestyle="--",
                   label=f"Global ({global_rate:.1%})")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Churn rate")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="x")

    for j in range(len(top_cats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Churn Rate by Categorical Variable\n"
                 "(Red = above average risk | Blue = below average risk)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "05_churn_by_categorical.png"))


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list,
                              output_dir: str) -> None:
    """Correlation heatmap for top numeric features."""
    corr_with_target = df[num_cols].corrwith(
        df[TARGET_COL].astype(float)
    ).abs().sort_values(ascending=False)
    top_cols = corr_with_target.head(20).index.tolist()

    corr_matrix = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot_kws={"size": 7},
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap — Top 20 Features by Correlation with Churn\n"
                 "(Lower triangle only — symmetry implied)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "06_correlation_heatmap.png"))


def plot_top_features_by_churn(df: pd.DataFrame, num_cols: list,
                                output_dir: str) -> None:
    """
    Boxplots of top 6 features by correlation with churn.
    Shows median, quartile ranges, and outliers for churners vs non-churners.
    """
    corr = df[num_cols].corrwith(df[TARGET_COL].astype(float)).abs()
    top_cols = corr.sort_values(ascending=False).head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(top_cols):
        ax = axes[i]
        data_0 = df[df[TARGET_COL] == 0][col].dropna()
        data_1 = df[df[TARGET_COL] == 1][col].dropna()

        bp = ax.boxplot(
            [data_0, data_1],
            patch_artist=True,
            labels=["No Churn", "Churn"],
            medianprops=dict(color="white", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(BLUE)
        bp["boxes"][1].set_facecolor(CORAL)
        for box in bp["boxes"]:
            box.set_alpha(0.7)

        # Mann-Whitney U test (non-parametric)
        _, p_val = stats.mannwhitneyu(data_0, data_1, alternative="two-sided")
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.set_title(f"{col}\np={p_val:.3f} {significance}", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Top 6 Features — Distribution by Churn\n"
                 "(Mann-Whitney U test: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "07_top_features_boxplot.png"))


# ---------------------------------------------------------------------------
# Main EDA function
# ---------------------------------------------------------------------------

def run_eda(data_path: str, output_dir: str) -> None:
    """
    Full exploratory data analysis pipeline.

    Args:
        data_path:  path to raw dataset.csv
        output_dir: directory where all EDA plots will be saved
    """
    logger.info("=" * 60)
    logger.info("EDA PIPELINE: Exploratory Data Analysis")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and fix
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, sep=";", low_memory=False)
    logger.info(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Drop ID columns
    id_cols = [c for c in df.columns
               if any(p in c for p in ID_PATTERNS)]
    df = df.drop(columns=id_cols, errors="ignore")

    # Fix decimal separators
    all_cat = CATEGORICAL_COLS + BINARY_CAT_COLS + [TARGET_COL]
    for col in df.columns:
        if col not in all_cat:
            df[col] = fix_decimal_separator(df[col])

    # ------------------------------------------------------------------
    # 2. Target distribution
    # ------------------------------------------------------------------
    logger.info("Plotting target distribution...")
    y = df[TARGET_COL].astype(int)
    logger.info(
        f"Churn=1: {y.sum():,} ({y.mean()*100:.1f}%) | "
        f"Churn=0: {(1-y).sum():,} ({(1-y.mean())*100:.1f}%)"
    )
    plot_target_distribution(y, output_dir)

    # ------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------
    logger.info("Plotting missing values...")
    plot_missing_values(df, output_dir)

    # ------------------------------------------------------------------
    # 4. Numeric distributions
    # ------------------------------------------------------------------
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_COL]
    logger.info(f"Plotting numeric distributions ({len(num_cols)} features)...")
    plot_numeric_distributions(df, num_cols, output_dir)

    # ------------------------------------------------------------------
    # 5. Outlier detection
    # ------------------------------------------------------------------
    logger.info("Detecting outliers...")
    plot_outlier_detection(df, num_cols, output_dir)

    # ------------------------------------------------------------------
    # 6. Churn rate by categorical
    # ------------------------------------------------------------------
    logger.info("Plotting churn rate by categorical variables...")
    plot_churn_by_categorical(df, output_dir)

    # ------------------------------------------------------------------
    # 7. Correlation heatmap
    # ------------------------------------------------------------------
    logger.info("Plotting correlation heatmap...")
    plot_correlation_heatmap(df, num_cols, output_dir)

    # ------------------------------------------------------------------
    # 8. Top features boxplot with Mann-Whitney test
    # ------------------------------------------------------------------
    logger.info("Plotting top features by churn...")
    plot_top_features_by_churn(df, num_cols, output_dir)

    # ------------------------------------------------------------------
    # 9. Summary statistics as JSON
    # ------------------------------------------------------------------
    logger.info("Computing summary statistics...")
    corr_with_target = df[num_cols].corrwith(
        df[TARGET_COL].astype(float)
    ).sort_values(key=abs, ascending=False)

    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)

    _dup_count = int(df.duplicated().sum())
    logger.info(f"Duplicate rows: {_dup_count:,}")

    _cat_cols_present = [c for c in CATEGORICAL_COLS + BINARY_CAT_COLS if c in df.columns]
    _cat_cardinality = {
        col: int(df[col].nunique(dropna=True))
        for col in _cat_cols_present
    }

    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "churn_rate": round(float(y.mean()), 4),
        "churn_count": int(y.sum()),
        "no_churn_count": int((1-y).sum()),
        "duplicate_row_count": _dup_count,
        "top_10_corr_with_churn": {
            col: round(float(val), 4)
            for col, val in corr_with_target.head(10).items()
        },
        "missing_pct": {
            col: float(val)
            for col, val in missing_pct[missing_pct > 0]
                               .sort_values(ascending=False).items()
        },
        "categorical_cardinality": _cat_cardinality,
    }

    json_path = os.path.join(output_dir, "eda_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {json_path}")

    logger.info("=" * 60)
    logger.info(f"EDA completed — {len(os.listdir(output_dir))} files saved to {output_dir}")
    logger.info("=" * 60)
