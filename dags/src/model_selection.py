"""
Task 3 — Model Selection
=========================
Loads both trained models (LR and LightGBM), compares them on the
identical hold-out test set (same random seed ensures same split),
and selects the best model.

Comparison strategy:
    1. Bootstrap AUC difference — non-parametric comparison of AUC-ROC.
       Resamples test set with replacement (n=500) and computes the
       empirical distribution of ΔAUC = AUC_LGBM - AUC_LR.
       H0 rejected if 0 is outside the 95% CI of the difference.
       Chosen over DeLong for robustness — no distributional assumptions.

    2. McNemar test — compares binary predictions at threshold 0.5.
       Tests whether models make errors on same or different samples.
       Note: threshold 0.5 is arbitrary for churn — used as secondary
       comparison only. Primary comparison is AUC-based (threshold-free).

Mathematical notes:
    - Bootstrap (Efron, 1979): empirical distribution of a statistic
      via resampling with replacement. No normality assumption required.
      For AUC difference: if 0 is outside [CI_low, CI_high] at 95%,
      we reject H0: AUC_1 = AUC_2.
    - McNemar (1947): chi-squared test on discordant predictions.
      H0: both models have same error rate.
      Uses Yates continuity correction for stability.
    - Model selection rule:
        * If ΔAUC CI includes 0 → no significant difference →
          prefer LR (Occam's razor: simpler, more interpretable)
        * If ΔAUC CI excludes 0 → significant difference →
          prefer model with higher AUC, unless business metrics
          (recall in particular) strongly favour the other
"""

import os
import json
import pickle
import logging
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, recall_score, precision_score

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_BOOTSTRAP = 500
CONFIDENCE = 0.95
SIGNIFICANCE_LEVEL = 0.05


# ---------------------------------------------------------------------------
# Bootstrap AUC difference
# ---------------------------------------------------------------------------

def bootstrap_auc_diff(y_true: np.ndarray,
                        y_prob_1: np.ndarray,
                        y_prob_2: np.ndarray,
                        name_1: str = "Model 1",
                        name_2: str = "Model 2") -> dict:
    """
    Bootstrap confidence interval for the difference in AUC-ROC.
    ΔAUC = AUC_model1 - AUC_model2

    H0: ΔAUC = 0 (models have equal AUC)
    H0 rejected if 0 is outside the (1-alpha) CI.

    Two-sided p-value approximation:
        p = 2 * min(P(ΔAUC <= 0), P(ΔAUC >= 0))
    Clamped to [0, 1].

    Args:
        y_true:   true binary labels
        y_prob_1: predicted probabilities from model 1
        y_prob_2: predicted probabilities from model 2
        name_1:   name of model 1
        name_2:   name of model 2

    Returns:
        dict with auc_1, auc_2, delta, ci, p_value, significant
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    diffs = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        # Skip if only one class in bootstrap sample
        if len(np.unique(yt)) < 2:
            continue
        try:
            a1 = roc_auc_score(yt, y_prob_1[idx])
            a2 = roc_auc_score(yt, y_prob_2[idx])
            diffs.append(a1 - a2)
        except Exception:
            continue

    if len(diffs) == 0:
        logger.error("Bootstrap produced no valid samples — cannot compare AUC")
        return {
            "method": "Bootstrap AUC difference",
            "auc_model_1": float(roc_auc_score(y_true, y_prob_1)),
            "auc_model_2": float(roc_auc_score(y_true, y_prob_2)),
            "delta_auc": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": 1.0,
            "significant": False,
            "error": "No valid bootstrap samples",
        }

    diffs = np.array(diffs)
    alpha = (1 - CONFIDENCE) / 2
    ci_low = float(np.percentile(diffs, alpha * 100))
    ci_high = float(np.percentile(diffs, (1 - alpha) * 100))

    # Two-sided p-value: probability that difference is on wrong side
    p_value = 2 * min(
        float(np.mean(diffs <= 0)),
        float(np.mean(diffs >= 0))
    )
    p_value = min(p_value, 1.0)
    p_value = max(p_value, 1.0 / N_BOOTSTRAP)

    auc_1 = float(roc_auc_score(y_true, y_prob_1))
    auc_2 = float(roc_auc_score(y_true, y_prob_2))

    return {
        "method": "Bootstrap AUC difference (n=500)",
        "model_1": name_1,
        "model_2": name_2,
        "auc_model_1": auc_1,
        "auc_model_2": auc_2,
        "delta_auc_mean": float(np.mean(diffs)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        # significant = CI does not contain 0 (primary criterion)
        # p_value < alpha is consistent but CI-based is more direct
        "significant": bool(ci_low > 0 or ci_high < 0),
        "error": None,
    }


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------

def mcnemar_test(y_true: np.ndarray,
                  y_pred_1: np.ndarray,
                  y_pred_2: np.ndarray,
                  name_1: str = "Model 1",
                  name_2: str = "Model 2") -> dict:
    """
    McNemar test on binary predictions at threshold 0.5.

    Contingency table of discordant predictions:
        b = model_1 correct, model_2 wrong
        c = model_1 wrong,   model_2 correct

    H0: b = c (both models have same error rate)
    Uses Yates continuity correction: chi2 = (|b-c| - 1)^2 / (b+c)

    Note: this is a SECONDARY comparison only.
    Threshold 0.5 is arbitrary for churn prediction.
    Primary comparison is AUC-based (threshold-free).
    """
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    b = int(np.sum(correct_1 & ~correct_2))
    c = int(np.sum(~correct_1 & correct_2))

    base = {
        "method": "McNemar (Yates continuity correction)",
        "note": "Secondary comparison — threshold 0.5 is arbitrary for churn",
        "model_1": name_1,
        "model_2": name_2,
        "b_model1_right_model2_wrong": b,
        "c_model1_wrong_model2_right": c,
    }

    if (b + c) == 0:
        return {
            **base,
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "note2": "No discordant pairs — models make identical predictions at threshold 0.5",
        }

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))

    return {
        **base,
        "chi2_statistic": float(chi2),
        "p_value": p_value,
        "significant": bool(p_value < SIGNIFICANCE_LEVEL),
    }


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------

def select_model(lr_model_path: str,
                  lgbm_model_path: str,
                  best_model_path: str,
                  output_dir: str) -> None:
    """
    Loads both models, compares them statistically, selects the best,
    and saves the winner as best_model.pkl.

    Args:
        lr_model_path:    path to lr_model.pkl
        lgbm_model_path:  path to lgbm_model.pkl
        best_model_path:  path where best model will be saved
        output_dir:       path where selection report JSON will be saved
    """
    logger.info("=" * 60)
    logger.info("TASK 3: Model Selection")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load both models
    # ------------------------------------------------------------------
    logger.info(f"Loading LR model from: {lr_model_path}")
    with open(lr_model_path, "rb") as f:
        lr_payload = pickle.load(f)

    logger.info(f"Loading LightGBM model from: {lgbm_model_path}")
    with open(lgbm_model_path, "rb") as f:
        lgbm_payload = pickle.load(f)

    # ------------------------------------------------------------------
    # 2. Extract predictions
    # ------------------------------------------------------------------
    y_test_lr   = np.array(lr_payload["y_test"])
    y_test_lgbm = np.array(lgbm_payload["y_test"])

    # Verify identical test sets (same seed guarantee)
    if len(y_test_lr) != len(y_test_lgbm):
        raise ValueError(
            f"Test set size mismatch: LR={len(y_test_lr)}, "
            f"LGBM={len(y_test_lgbm)}. Check random seeds."
        )
    if not np.array_equal(y_test_lr, y_test_lgbm):
        raise ValueError(
            "Test set labels differ between models — "
            "models were trained on different splits!"
        )
    logger.info("✓ Verified: both models evaluated on identical test set")

    y_true      = y_test_lr
    y_prob_lr   = np.array(lr_payload["y_prob"])
    y_prob_lgbm = np.array(lgbm_payload["y_prob"])
    y_pred_lr   = np.array(lr_payload["y_pred"])
    y_pred_lgbm = np.array(lgbm_payload["y_pred"])

    # ------------------------------------------------------------------
    # 3. Metrics comparison table
    # ------------------------------------------------------------------
    # Use intersection of available metrics — robust to missing keys
    common_metrics = sorted(
        set(lr_payload["holdout_metrics"]) &
        set(lgbm_payload["holdout_metrics"])
    )

    logger.info("--- Metrics Comparison (Hold-out Test Set) ---")
    logger.info(f"  {'Metric':12s}  {'LR':>10s}  {'LightGBM':>10s}  {'Winner':>12s}")
    logger.info(f"  {'-'*52}")

    winner_counts = {"Logistic Regression": 0, "LightGBM": 0}
    for metric in common_metrics:
        lr_val   = lr_payload["holdout_metrics"][metric]["mean"]
        lgbm_val = lgbm_payload["holdout_metrics"][metric]["mean"]
        winner   = "LightGBM" if lgbm_val > lr_val else "Logistic Regression"
        winner_counts[winner] += 1
        logger.info(
            f"  {metric:12s}  {lr_val:>10.4f}  {lgbm_val:>10.4f}  "
            f"← {winner}"
        )

    # ------------------------------------------------------------------
    # 4. Bootstrap AUC difference (primary statistical comparison)
    # ------------------------------------------------------------------
    logger.info("--- Primary Statistical Comparison: Bootstrap AUC Difference ---")
    bootstrap_result = bootstrap_auc_diff(
        y_true, y_prob_lgbm, y_prob_lr,
        name_1="LightGBM", name_2="Logistic Regression"
    )

    logger.info(f"  Method        : {bootstrap_result['method']}")
    logger.info(f"  AUC LightGBM  : {bootstrap_result['auc_model_1']:.4f}")
    logger.info(f"  AUC LR        : {bootstrap_result['auc_model_2']:.4f}")
    _delta_mean = bootstrap_result.get('delta_auc_mean')
    _ci_low     = bootstrap_result.get('ci_low')
    _ci_high    = bootstrap_result.get('ci_high')
    logger.info(f"  Mean ΔAUC     : {_delta_mean:.4f}" if _delta_mean is not None else "  Mean ΔAUC     : N/A")
    if _ci_low is not None and _ci_high is not None:
        logger.info(f"  95% CI        : [{_ci_low:.4f}, {_ci_high:.4f}]")
    else:
        logger.info("  95% CI        : N/A (bootstrap failed)")
    logger.info(f"  p-value       : {bootstrap_result['p_value']:.4f}")
    logger.info(
        f"  Significant   : {bootstrap_result['significant']} "
        f"(α = {SIGNIFICANCE_LEVEL})"
    )

    # ------------------------------------------------------------------
    # 5. McNemar test (secondary comparison)
    # ------------------------------------------------------------------
    logger.info("--- Secondary Comparison: McNemar Test (threshold = 0.5) ---")
    mcnemar_result = mcnemar_test(
        y_true, y_pred_lgbm, y_pred_lr,
        name_1="LightGBM", name_2="Logistic Regression"
    )
    logger.info(f"  LGBM right / LR wrong : {mcnemar_result['b_model1_right_model2_wrong']}")
    logger.info(f"  LGBM wrong / LR right : {mcnemar_result['c_model1_wrong_model2_right']}")
    logger.info(f"  chi2                  : {mcnemar_result.get('chi2_statistic', 'N/A')}")
    logger.info(f"  p-value               : {mcnemar_result['p_value']:.4f}")
    logger.info(f"  Significant           : {mcnemar_result['significant']}")
    logger.info(f"  Note                  : {mcnemar_result['note']}")

    # McNemar test at each model's own business threshold
    lr_biz_thr   = lr_payload.get("thresholds", {}).get("business_threshold", 0.5)
    lgbm_biz_thr = lgbm_payload.get("thresholds", {}).get("business_threshold", 0.5)
    y_pred_lr_business   = (y_prob_lr   >= lr_biz_thr).astype(int)
    y_pred_lgbm_business = (y_prob_lgbm >= lgbm_biz_thr).astype(int)
    logger.info(
        f"--- McNemar Test (business thresholds: LR={lr_biz_thr:.3f}, "
        f"LGBM={lgbm_biz_thr:.3f}) ---"
    )
    mcnemar_result_business = mcnemar_test(
        y_true, y_pred_lgbm_business, y_pred_lr_business,
        name_1="LightGBM", name_2="Logistic Regression"
    )
    logger.info(f"  LGBM right / LR wrong : {mcnemar_result_business['b_model1_right_model2_wrong']}")
    logger.info(f"  LGBM wrong / LR right : {mcnemar_result_business['c_model1_wrong_model2_right']}")
    logger.info(f"  chi2                  : {mcnemar_result_business.get('chi2_statistic', 'N/A')}")
    logger.info(f"  p-value               : {mcnemar_result_business['p_value']:.4f}")
    logger.info(f"  Significant           : {mcnemar_result_business['significant']}")

    # ------------------------------------------------------------------
    # 6. Business metrics check (recall — most important for churn)
    # ------------------------------------------------------------------
    recall_lgbm = lgbm_payload["holdout_metrics"].get("Recall", {}).get("mean", 0)
    recall_lr   = lr_payload["holdout_metrics"].get("Recall", {}).get("mean", 0)
    recall_delta = recall_lgbm - recall_lr
    logger.info(f"--- Business Metrics Check ---")
    logger.info(f"  Recall LGBM : {recall_lgbm:.4f}")
    logger.info(f"  Recall LR   : {recall_lr:.4f}")
    logger.info(f"  Δ Recall    : {recall_delta:+.4f}")

    # ------------------------------------------------------------------
    # 7. Final decision
    # ------------------------------------------------------------------
    auc_lgbm = bootstrap_result["auc_model_1"]
    auc_lr   = bootstrap_result["auc_model_2"]
    significant = bootstrap_result["significant"]

    if not significant:
        # No significant AUC difference → prefer simpler model
        # Unless recall is substantially better for LGBM (>3pp)
        if recall_delta > 0.03:
            best_name = "LightGBM"
            best_payload = lgbm_payload
            reason = (
                f"AUC difference not statistically significant "
                f"(p={bootstrap_result['p_value']:.3f}), but LightGBM "
                f"shows meaningfully higher recall "
                f"(+{recall_delta*100:.1f}pp) — preferred for churn "
                f"where catching churners matters most."
            )
        else:
            best_name = "Logistic Regression"
            best_payload = lr_payload
            reason = (
                f"No statistically significant AUC difference "
                f"(p={bootstrap_result['p_value']:.3f} > {SIGNIFICANCE_LEVEL}, "
                f"95% CI includes 0). "
                f"Recall difference is negligible ({recall_delta*100:+.1f}pp). "
                f"Preferring simpler model (Logistic Regression) — Occam's razor."
            )
    else:
        if auc_lgbm >= auc_lr:
            best_name = "LightGBM"
            best_payload = lgbm_payload
            reason = (
                f"LightGBM significantly outperforms LR in AUC "
                f"(ΔAUC = {auc_lgbm - auc_lr:.4f}, "
                f"p={bootstrap_result['p_value']:.4f} < {SIGNIFICANCE_LEVEL}, "
                f"95% CI: [{bootstrap_result['ci_low']:.4f}, "
                f"{bootstrap_result['ci_high']:.4f}])."
            )
        else:
            best_name = "Logistic Regression"
            best_payload = lr_payload
            reason = (
                f"LR significantly outperforms LightGBM in AUC "
                f"(ΔAUC = {auc_lr - auc_lgbm:.4f}, "
                f"p={bootstrap_result['p_value']:.4f} < {SIGNIFICANCE_LEVEL})."
            )

    logger.info(f"--- WINNER: {best_name} ---")
    logger.info(f"  Reason: {reason}")

    # ------------------------------------------------------------------
    # 8. Save best model
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # Extract actual LR coefficients (not the full model object)
    try:
        lr_pipeline = lr_payload["model"]
        lr_coef_values = lr_pipeline.named_steps["clf"].coef_[0].tolist()
        lr_coef_names  = lr_payload["feature_names"]
    except Exception:
        lr_coef_values = None
        lr_coef_names  = None

    best_payload["selected_model_name"] = best_name
    best_payload["selection_reason"]    = reason
    best_payload["bootstrap_result"]    = bootstrap_result
    best_payload["mcnemar_result"]      = mcnemar_result
    best_payload["mcnemar_test_threshold_0_5"]      = mcnemar_result
    best_payload["mcnemar_test_business_threshold"] = mcnemar_result_business
    best_payload["lr_metrics"]          = lr_payload["holdout_metrics"]
    best_payload["lgbm_metrics"]        = lgbm_payload["holdout_metrics"]
    best_payload["lr_coefficients"]     = lr_coef_values
    best_payload["lr_coefficient_names"] = lr_coef_names
    best_payload["lr_thresholds"]        = lr_payload.get("thresholds")
    best_payload["lgbm_thresholds"]      = lgbm_payload.get("thresholds")

    with open(best_model_path, "wb") as f:
        pickle.dump(best_payload, f)
    logger.info(f"Best model saved to: {best_model_path}")

    # ------------------------------------------------------------------
    # 9. Save selection report as JSON
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "winner": best_name,
        "reason": reason,
        "bootstrap_auc_comparison": bootstrap_result,
        "mcnemar_test": mcnemar_result,
        "mcnemar_test_threshold_0_5": mcnemar_result,
        "mcnemar_test_business_threshold": mcnemar_result_business,
        "lr_thresholds": lr_payload.get("thresholds"),
        "lgbm_thresholds": lgbm_payload.get("thresholds"),
        "metrics_comparison": {
            "logistic_regression": {
                k: round(v["mean"], 4)
                for k, v in lr_payload["holdout_metrics"].items()
            },
            "lightgbm": {
                k: round(v["mean"], 4)
                for k, v in lgbm_payload["holdout_metrics"].items()
            },
        },
    }
    report_path = os.path.join(output_dir, "model_selection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Selection report saved to: {report_path}")

    logger.info("=" * 60)
    logger.info(f"TASK 3 completed — Winner: {best_name}")
    logger.info("=" * 60)
