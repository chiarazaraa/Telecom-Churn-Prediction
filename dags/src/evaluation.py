"""
Task 4 — Final Evaluation + SHAP Explainability
=================================================
Loads the best model (selected in Task 3) and produces:

    1. Full metrics on hold-out test set with bootstrap CI
    2. ROC curve + Precision-Recall curve
    3. Calibration curve (display only — no post-hoc calibration)
    4. Confusion matrix
    5. SHAP explainability:
         - If LightGBM won: TreeExplainer (exact, fast)
         - If LR won: coefficients + LinearExplainer
    6. SHAP vs LR coefficients comparison (Spearman rank correlation)
    7. All plots saved as PNG in outputs/
    8. Metrics saved as JSON

Mathematical notes:
    - SHAP (Lundberg & Lee, 2017): Shapley values from cooperative
      game theory (Shapley, 1953). Each feature's SHAP value =
      weighted average of its marginal contribution across all
      possible feature coalitions.
    - TreeExplainer: exact Shapley values for tree ensembles.
    - LinearExplainer: exact Shapley values for linear models.
      SHAP_i = beta_i * (x_i - E[x_i])
    - Calibration ECE (approximate, unweighted): mean |predicted - actual|
      across bins. True ECE weights by bin size — noted as approximation.
    - Standard metrics and classification_report use threshold 0.5.
    - Additional business metrics (confusion matrix, P/R/F1) are computed at
      the validation-selected business threshold from the model payload.
    - Threshold curves computed on the test set are diagnostic only; threshold
      selection was performed on an internal validation split (no test leakage).
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
)
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

RANDOM_STATE       = 42
N_BOOTSTRAP        = 300
N_BOOTSTRAP_CURVE  = 100   # smaller n for CI bands on plots — keeps runtime reasonable
CONFIDENCE         = 0.95
SHAP_SAMPLE_SIZE   = 2000

BLUE   = "#2563eb"
PURPLE = "#7c3aed"
GREEN  = "#059669"
CORAL  = "#dc2626"
GRAY   = "#6b7280"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division — returns default if denominator is zero."""
    return float(num / den) if den != 0 else default


def bootstrap_metric(y_true, y_pred, y_prob, metric_fn,
                     n_iter=N_BOOTSTRAP, confidence=CONFIDENCE):
    """Bootstrap CI via percentile method (Efron, 1979)."""
    rng = np.random.RandomState(RANDOM_STATE)
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx], y_prob[idx]))
        except Exception:
            pass
    if len(scores) == 0:
        return (float(metric_fn(y_true, y_pred, y_prob)), 0.0, 1.0)
    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    return (float(np.mean(scores)),
            float(np.percentile(scores, alpha * 100)),
            float(np.percentile(scores, (1 - alpha) * 100)))


def extract_shap_values(shap_explanation) -> np.ndarray:
    """
    Safely extracts 2D SHAP values for the positive class.
    Handles both old (list) and new (3D array) SHAP API formats.
    """
    values = shap_explanation.values
    if isinstance(values, list):
        # Old SHAP API: list of arrays per class
        return values[1]
    elif values.ndim == 3:
        # New SHAP API: (n_samples, n_features, n_classes)
        return values[:, :, 1]
    else:
        # Already 2D
        return values


def get_base_estimator(model):
    """
    Extracts the base estimator from a sklearn Pipeline if needed.
    Returns (estimator, is_pipeline).
    """
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        return model.named_steps[list(model.named_steps.keys())[-1]], True
    return model, False


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true, y_prob, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Bootstrap CI bands on fixed FPR grid
    _fpr_grid  = np.linspace(0, 1, 100)
    _tpr_boot  = []
    _rng_roc   = np.random.RandomState(RANDOM_STATE)
    _yt_arr    = np.array(y_true)
    _yp_arr    = np.array(y_prob)
    _n_roc     = len(_yt_arr)
    try:
        for _ in range(N_BOOTSTRAP_CURVE):
            _idx = _rng_roc.randint(0, _n_roc, size=_n_roc)
            if len(np.unique(_yt_arr[_idx])) < 2:
                continue
            _fpr_b, _tpr_b, _ = roc_curve(_yt_arr[_idx], _yp_arr[_idx])
            _tpr_boot.append(np.interp(_fpr_grid, _fpr_b, _tpr_b))
    except Exception:
        _tpr_boot = []

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(_tpr_boot) >= 10:
        _tpr_arr = np.array(_tpr_boot)
        ax.fill_between(
            _fpr_grid,
            np.percentile(_tpr_arr, 2.5, axis=0),
            np.percentile(_tpr_arr, 97.5, axis=0),
            alpha=0.2, color=BLUE,
            label=f"95% CI (n={len(_tpr_boot)} bootstrap)",
        )
    ax.plot(fpr, tpr, color=BLUE, lw=2,
            label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle="--",
            label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: roc_curve.png (AUC = {auc:.4f})")


def plot_precision_recall(y_true, y_prob, model_name, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    # Bootstrap CI bands on fixed recall grid
    # precision_recall_curve returns decreasing recall; reverse for interp
    _rec_grid  = np.linspace(0, 1, 100)
    _prec_boot = []
    _rng_pr    = np.random.RandomState(RANDOM_STATE)
    _yt_arr    = np.array(y_true)
    _yp_arr    = np.array(y_prob)
    _n_pr      = len(_yt_arr)
    try:
        for _ in range(N_BOOTSTRAP_CURVE):
            _idx = _rng_pr.randint(0, _n_pr, size=_n_pr)
            if len(np.unique(_yt_arr[_idx])) < 2:
                continue
            _prec_b, _rec_b, _ = precision_recall_curve(_yt_arr[_idx], _yp_arr[_idx])
            # Reverse so recall is ascending; deduplicate for np.interp
            _prec_b_r = _prec_b[::-1]
            _rec_b_r  = _rec_b[::-1]
            _uniq_rec, _uniq_idx = np.unique(_rec_b_r, return_index=True)
            _prec_boot.append(np.interp(_rec_grid, _uniq_rec, _prec_b_r[_uniq_idx]))
    except Exception:
        _prec_boot = []

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(_prec_boot) >= 10:
        _prec_arr = np.array(_prec_boot)
        ax.fill_between(
            _rec_grid,
            np.percentile(_prec_arr, 2.5, axis=0),
            np.percentile(_prec_arr, 97.5, axis=0),
            alpha=0.2, color=PURPLE,
            label=f"95% CI (n={len(_prec_boot)} bootstrap)",
        )
    ax.plot(recall, precision, color=PURPLE, lw=2,
            label=f"{model_name} (AP = {ap:.4f})")
    ax.axhline(baseline, color=GRAY, lw=1, linestyle="--",
               label=f"Random baseline (AP = {baseline:.2f})")
    ax.fill_between(recall, precision, alpha=0.1, color=PURPLE)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: precision_recall_curve.png (AP = {ap:.4f})")


def plot_calibration(y_true, y_prob, model_name, output_dir):
    """
    Calibration curve — display only, no post-hoc calibration on test set.
    ECE reported as unweighted approximation (mean |predicted - actual|).
    True ECE weights each bin by its sample count.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    approx_ece = float(np.mean(np.abs(prob_true - prob_pred)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, color=GREEN, lw=2, marker="o",
            label=f"{model_name} (approx ECE = {approx_ece:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle="--",
            label="Perfect calibration")
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color=GREEN)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual churn rate)")
    ax.set_title(f"Calibration Curve — {model_name}\n"
                 "(approx ECE — unweighted mean absolute deviation across bins)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: calibration_curve.png (approx ECE = {approx_ece:.3f})")

    if approx_ece > 0.05:
        logger.warning(
            f"Approx ECE = {approx_ece:.3f} > 0.05. "
            "Future improvement: apply CalibratedClassifierCV on a "
            "dedicated validation set (never on the test set)."
        )
    return approx_ece


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir,
                           filename="confusion_matrix.png",
                           subtitle="threshold = 0.5"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)
    classes = ["No Churn", "Churn"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix — {model_name}\n({subtitle})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {filename}")
    return cm


def plot_shap_summary(shap_values, X_plot, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_plot, show=False, max_display=20)
    plt.title(
        "SHAP Summary Plot — Global Feature Impact on Churn\n"
        "(Shapley values: average marginal contribution across all coalitions)",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_summary.png")


def plot_shap_bar(shap_values, X_plot, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_plot, plot_type="bar",
                      show=False, max_display=20)
    plt.title(
        "SHAP Feature Importance — Mean |SHAP value|\n"
        "(More rigorous than Gini: accounts for feature interactions)",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_bar.png")


def plot_shap_waterfall(shap_explanation, output_dir, customer_idx=0):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_explanation[customer_idx], show=False)
    plt.title(
        f"SHAP Waterfall — Individual Explanation (Customer #{customer_idx})\n"
        "Red = increases churn risk | Blue = decreases churn risk",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: shap_waterfall.png")


def plot_shap_vs_lr(mean_abs_shap, shap_feature_names,
                     lr_model_path, output_dir):
    """
    Compares SHAP importance with LR standardised coefficients.
    Uses Spearman rank correlation to quantify agreement.
    """
    try:
        with open(lr_model_path, "rb") as f:
            lr_payload = pickle.load(f)

        lr_features = lr_payload.get("feature_names", [])
        lr_pipeline  = lr_payload.get("model")
        if lr_pipeline is None or not lr_features:
            logger.warning("LR model or feature names not found — skipping comparison")
            return

        lr_coefs = lr_pipeline.named_steps["clf"].coef_[0]
        lr_abs   = np.abs(lr_coefs)

        # Common features between SHAP and LR
        common = [f for f in shap_feature_names if f in lr_features]
        if len(common) < 5:
            logger.warning(f"Only {len(common)} common features — skipping SHAP vs LR plot")
            return

        shap_vals = np.array([
            mean_abs_shap[shap_feature_names.index(f)] for f in common
        ])
        lr_vals = np.array([
            lr_abs[lr_features.index(f)] for f in common
        ])

        # Normalise to [0, 1]
        shap_norm = shap_vals / (shap_vals.max() + 1e-10)
        lr_norm   = lr_vals   / (lr_vals.max()   + 1e-10)

        # Spearman rank correlation
        rho, p_val = spearmanr(shap_norm, lr_norm)
        logger.info(
            f"Spearman ρ (SHAP vs LR): {rho:.3f} (p={p_val:.3f}) — "
            + ("strong agreement: result robust & model-agnostic"
               if rho > 0.7 else
               "disagreement: LightGBM captures non-linearities LR cannot model")
        )

        # Plot
        x = np.arange(len(common))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, shap_norm, width, label="SHAP |value| (normalised)",
               color=BLUE, alpha=0.8)
        ax.bar(x + width/2, lr_norm,   width, label="|LR coefficient| (normalised)",
               color=CORAL, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Normalised importance")
        ax.set_title(
            f"SHAP vs Logistic Regression — Feature Importance Comparison\n"
            f"Spearman ρ = {rho:.3f} — "
            + ("Agreement: robust result" if rho > 0.7
               else "Disagreement: non-linearities matter")
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "shap_vs_lr.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved: shap_vs_lr.png")

    except Exception as e:
        logger.warning(f"SHAP vs LR comparison failed: {e}")


# ---------------------------------------------------------------------------
# Threshold optimization & business analysis
# ---------------------------------------------------------------------------

def find_optimal_thresholds(y_true, y_prob):
    thresholds = np.linspace(0, 1, 200)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    best_f1_idx = np.argmax(f1s)
    best_f1_threshold = float(thresholds[best_f1_idx])
    business_candidates_mask = precisions >= 0.60
    if business_candidates_mask.any():
        business_idx = np.where(business_candidates_mask)[0][np.argmax(recalls[business_candidates_mask])]
        business_threshold = float(thresholds[business_idx])
        idx = np.argmin(np.abs(thresholds - business_threshold))
        business_recall = float(recalls[idx])
    else:
        business_threshold = 0.5
        business_recall = float(recalls[np.argmin(np.abs(thresholds - 0.5))])
    return {
        'threshold_0_5': 0.5,
        'best_f1_threshold': best_f1_threshold,
        'business_threshold': business_threshold,
        'best_f1': float(f1s[best_f1_idx]),
        'business_recall': business_recall,
    }, thresholds, precisions, recalls, f1s


def plot_threshold_curve(thresholds, precisions, recalls, f1s, thresholds_dict, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, color='#2563eb', lw=2, label='Precision')
    ax.plot(thresholds, recalls, color='#dc2626', lw=2, label='Recall')
    ax.plot(thresholds, f1s, color='#059669', lw=2, label='F1 Score')
    ax.axvline(thresholds_dict['threshold_0_5'], color='gray', lw=1.5,
               linestyle='--', label='Threshold 0.5')
    ax.axvline(thresholds_dict['best_f1_threshold'], color='#059669', lw=1.5,
               linestyle=':', label=f"Best F1 ({thresholds_dict['best_f1_threshold']:.2f})")
    ax.axvline(thresholds_dict['business_threshold'], color='#dc2626', lw=1.5,
               linestyle=':', label=f"Business ({thresholds_dict['business_threshold']:.2f})")
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision / Recall / F1 vs Threshold\n(Business threshold = max recall with precision >= 0.60)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Saving precision_recall_vs_threshold.png to {output_dir}')
    fig.savefig(os.path.join(output_dir, 'precision_recall_vs_threshold.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved: precision_recall_vs_threshold.png')


def plot_business_segmentation(y_true, y_prob, output_dir):
    df_seg = pd.DataFrame({'y_true': np.array(y_true), 'y_prob': np.array(y_prob)})
    df_seg = df_seg.sort_values('y_prob', ascending=False).reset_index(drop=True)
    n = len(df_seg)
    total_churners = max(int(np.array(y_true).sum()), 1)
    segments = {}
    for pct in [10, 20, 30]:
        top_n = int(n * pct / 100)
        segment = df_seg.iloc[:top_n]
        churn_rate = float(segment['y_true'].mean())
        churn_captured = float(segment['y_true'].sum() / total_churners * 100)
        segments[f'top_{pct}_percent'] = {
            'n_customers': top_n,
            'churn_rate_pct': round(churn_rate * 100, 1),
            'churn_captured_pct': round(churn_captured, 1),
        }
        logger.info(f"Top {pct}%: churn rate={churn_rate*100:.1f}%, churners captured={churn_captured:.1f}%")
    cumulative_pct = np.arange(1, n + 1) / n * 100
    cumulative_churn = df_seg['y_true'].cumsum() / total_churners * 100

    # Bootstrap CI for gain curve — resample test set with replacement
    _gain_boot  = []
    _rng_gc     = np.random.RandomState(RANDOM_STATE)
    _y_true_gc  = np.array(y_true)
    _y_prob_gc  = np.array(y_prob)
    try:
        for _ in range(N_BOOTSTRAP_CURVE):
            _idx      = _rng_gc.randint(0, n, size=n)
            _bt       = _y_true_gc[_idx]
            _bp       = _y_prob_gc[_idx]
            _total_ch = max(int(_bt.sum()), 1)
            _order    = np.argsort(_bp)[::-1]
            _gain_boot.append(np.cumsum(_bt[_order]) / _total_ch * 100)
    except Exception:
        _gain_boot = []

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(_gain_boot) >= 10:
        _gc_arr = np.array(_gain_boot)
        ax.fill_between(
            cumulative_pct,
            np.percentile(_gc_arr, 2.5, axis=0),
            np.percentile(_gc_arr, 97.5, axis=0),
            alpha=0.2, color='#2563eb',
            label=f'95% CI (n={len(_gain_boot)} bootstrap)',
        )
    ax.plot(cumulative_pct, cumulative_churn, color='#2563eb', lw=2, label='Model')
    ax.plot([0, 100], [0, 100], color='gray', lw=1.5, linestyle='--', label='Random baseline')
    for pct in [10, 20, 30]:
        top_n = int(n * pct / 100)
        captured = float(df_seg['y_true'].iloc[:top_n].sum() / total_churners * 100)
        ax.annotate(f'Top {pct}%\n{captured:.0f}% churners',
                    xy=(pct, captured), xytext=(pct + 3, captured - 8),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))
    ax.set_xlabel('% of customers contacted (ranked by churn probability)')
    ax.set_ylabel('% of churners captured')
    ax.set_title('Gain Curve — Business Value of the Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Saving gain_curve.png to {output_dir}')
    fig.savefig(os.path.join(output_dir, 'gain_curve.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved: gain_curve.png')

    # Sampled gain curve data for JSON export (~200 points)
    step = max(1, n // 200)
    gain_curve_data = [
        {
            "customer_pct": round(float(cumulative_pct[i]), 2),
            "churners_captured_pct": round(float(cumulative_churn.iloc[i]), 2),
        }
        for i in range(0, n, step)
    ]
    segments["gain_curve_data"] = gain_curve_data
    return segments


def export_predictions(y_test, y_prob, thresholds_dict, output_dir):
    df_pred = pd.DataFrame({
        'churn_probability': np.array(y_prob),
        'prediction_0_5': (np.array(y_prob) >= 0.5).astype(int),
        'prediction_best_f1': (np.array(y_prob) >= thresholds_dict['best_f1_threshold']).astype(int),
        'prediction_business': (np.array(y_prob) >= thresholds_dict['business_threshold']).astype(int),
        'true_label': np.array(y_test),
    })
    df_pred = df_pred.sort_values('churn_probability', ascending=False).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'predictions.csv')
    logger.info(f'Saving predictions.csv to {output_dir}')
    df_pred.to_csv(path, index=False)
    logger.info(f'Saved: predictions.csv ({len(df_pred)} rows)')
    logger.info(f'Predictions saved: {len(df_pred)} rows')
    return df_pred


def export_top_shap_drivers(shap_values, feature_names, y_prob, output_dir,
                             top_n_customers=100, top_n_features=3):
    if shap_values is None:
        logger.warning('SHAP values not available — skipping top drivers export')
        return
    y_prob_arr = np.array(y_prob)
    if len(y_prob_arr) != len(shap_values):
        logger.warning(
            f'y_prob length ({len(y_prob_arr)}) != shap_values length ({len(shap_values)}) '
            f'— skipping top drivers export (pass y_prob_sample, not full y_prob)'
        )
        return
    top_idx = np.argsort(y_prob_arr)[::-1][:top_n_customers]
    drivers = []
    for rank, idx in enumerate(top_idx):
        customer_shap = shap_values[idx]
        top_feature_idx = np.argsort(np.abs(customer_shap))[::-1][:top_n_features]
        drivers.append({
            'rank': rank + 1,
            'churn_probability': round(float(y_prob_arr[idx]), 4),
            'top_drivers': [
                {
                    'feature': feature_names[fi],
                    'shap_value': round(float(customer_shap[fi]), 4),
                    'direction': 'increases_churn' if customer_shap[fi] > 0 else 'decreases_churn',
                }
                for fi in top_feature_idx
            ]
        })
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'top_churn_drivers.json')
    logger.info(f'Saving top_churn_drivers.json to {output_dir}')
    with open(path, 'w') as f:
        json.dump(drivers, f, indent=2)
    logger.info(f'Saved: top_churn_drivers.json ({top_n_customers} high-risk customers)')
    return drivers


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(best_model_path: str,
                   lr_model_path: str,
                   output_dir: str) -> None:
    """
    Full evaluation of the best model with SHAP explainability.

    Args:
        best_model_path: path to best_model.pkl
        lr_model_path:   path to lr_model.pkl (for SHAP vs LR comparison)
        output_dir:      directory for all outputs
    """
    logger.info("=" * 60)
    logger.info("TASK 4: Final Evaluation + SHAP Explainability")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    with open(best_model_path, "rb") as f:
        payload = pickle.load(f)

    model        = payload["model"]
    model_name   = payload["selected_model_name"]
    feature_names = payload["feature_names"]
    X_test       = payload["X_test"]
    y_test       = np.array(payload["y_test"])
    y_prob       = np.array(payload["y_prob"])
    y_pred       = np.array(payload["y_pred"])

    logger.info(f"Best model: {model_name}")
    logger.info(f"Test set: {len(y_test):,} samples")
    os.makedirs(output_dir, exist_ok=True)

    # Validate feature_names length consistency
    if len(feature_names) != X_test.shape[1]:
        logger.warning(
            f"feature_names length ({len(feature_names)}) != "
            f"X_test columns ({X_test.shape[1]}) — using X_test columns"
        )
        feature_names = list(X_test.columns)

    # ------------------------------------------------------------------
    # 2. Metrics with bootstrap CI
    # ------------------------------------------------------------------
    logger.info("--- Final Metrics (Bootstrap CI 95%, n=300, threshold=0.5) ---")
    metrics_fns = {
        "AUC-ROC":   lambda yt, yp, ypr: roc_auc_score(yt, ypr),
        "F1":        lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
        "Precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
        "Recall":    lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        "Avg Prec":  lambda yt, yp, ypr: average_precision_score(yt, ypr),
    }
    final_metrics = {}
    for name, fn in metrics_fns.items():
        mean, lo, hi = bootstrap_metric(y_test, y_pred, y_prob, fn)
        final_metrics[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        logger.info(f"  {name:12s}: {mean:.4f}  [95% CI: {lo:.4f} - {hi:.4f}]")

    # ------------------------------------------------------------------
    # 3. Classification report
    # ------------------------------------------------------------------
    logger.info("--- Classification Report (threshold = 0.5) ---")
    for line in classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"]
    ).split("\n"):
        if line.strip():
            logger.info(f"  {line}")

    # ------------------------------------------------------------------
    # 4. Business interpretation (threshold = 0.5 — stated explicitly)
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn_count, tp = cm.ravel()
    catch_rate = safe_div(tp, tp + fn_count)
    precision_val = safe_div(tp, tp + fp)

    logger.info("--- Business Interpretation (threshold = 0.5) ---")
    logger.info(f"  Churners correctly identified (TP): {tp:,}")
    logger.info(f"  Churners missed (FN):               {fn_count:,}")
    logger.info(f"  False alarms (FP):                  {fp:,}")
    logger.info(f"  Churn catch rate (Recall):          {catch_rate*100:.1f}%")
    logger.info(
        f"  Of customers flagged as churners: "
        f"{precision_val*100:.1f}% actually churn (Precision)"
    )

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    plot_roc_curve(y_test, y_prob, model_name, output_dir)
    plot_precision_recall(y_test, y_prob, model_name, output_dir)
    approx_ece = plot_calibration(y_test, y_prob, model_name, output_dir)

    # Weighted ECE: true ECE definition, weighting each bin by its sample count
    _ece_edges = np.linspace(0.0, 1.0, 11)
    _ece_edges[-1] += 1e-8
    _bin_idx = np.clip(np.digitize(y_prob, _ece_edges) - 1, 0, 9)
    _n_total = len(y_prob)
    weighted_ece = float(sum(
        (np.sum(_bin_idx == i) / _n_total) * abs(
            float(y_prob[_bin_idx == i].mean()) - float(y_test[_bin_idx == i].mean())
        )
        for i in range(10)
        if np.sum(_bin_idx == i) > 0
    ))
    logger.info(f"Weighted ECE (bin-count weighted): {weighted_ece:.4f}")

    plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

    # ------------------------------------------------------------------
    # 6. SHAP explainability
    # ------------------------------------------------------------------
    logger.info(f"Computing SHAP values for {model_name}...")

    rng = np.random.RandomState(RANDOM_STATE)
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
    sample_idx    = rng.choice(len(X_test), size=sample_size, replace=False)
    X_sample      = X_test.iloc[sample_idx].copy()
    y_prob_sample = y_prob[sample_idx]

    is_lgbm = "LightGBM" in model_name
    is_lr   = "Logistic Regression" in model_name

    shap_values      = None
    shap_explanation = None
    mean_abs_shap    = None
    X_plot           = X_sample  # default — may be overridden for LR

    if is_lgbm:
        logger.info("TreeExplainer — exact Shapley values for tree ensembles")
        # SHAP values reflect model score (log-odds) contributions, not direct probability % changes.
        try:
            # Extract base estimator if model is a Pipeline
            base_model, is_pipeline = get_base_estimator(model)
            explainer        = shap.TreeExplainer(base_model)
            shap_explanation = explainer(X_sample)
            shap_values      = extract_shap_values(shap_explanation)
            X_plot           = X_sample
        except Exception as e:
            logger.error(f"TreeExplainer failed: {e}")

    elif is_lr:
        logger.info("LinearExplainer — exact Shapley values for linear models")
        logger.info("Note: SHAP_i = beta_i * (x_i - E[x_i])")
        try:
            # Transform X_sample with all pipeline steps except clf
            X_transformed_arr = model[:-1].transform(X_sample)
            # Keep as DataFrame with correct feature names for SHAP plots
            X_transformed_df = pd.DataFrame(
                X_transformed_arr,
                columns=feature_names,
                index=X_sample.index,
            )
            clf = model.named_steps["clf"]
            explainer        = shap.LinearExplainer(clf, X_transformed_df)
            shap_explanation = explainer(X_transformed_df)
            shap_values      = extract_shap_values(shap_explanation)
            X_plot           = X_transformed_df  # use transformed for plots
        except Exception as e:
            logger.warning(f"LinearExplainer failed: {e} — using coefficients only")

    # SHAP results
    if shap_values is not None:
        # Validate shape
        if shap_values.shape[1] != len(feature_names):
            logger.warning(
                f"SHAP values shape {shap_values.shape} != "
                f"feature_names length {len(feature_names)} — skipping SHAP plots"
            )
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_n  = 15
            top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

            logger.info(f"--- Top {top_n} Features by mean |SHAP value| ---")
            for rank, idx in enumerate(top_idx, 1):
                direction = (
                    "↑ increases" if np.mean(shap_values[:, idx]) > 0
                    else "↓ decreases"
                )
                logger.info(
                    f"  {rank:2d}. {feature_names[idx]:45s}: "
                    f"mean |SHAP| = {mean_abs_shap[idx]:.4f}  "
                    f"({direction} churn risk)"
                )

            plot_shap_summary(shap_values, X_plot, output_dir)
            plot_shap_bar(shap_values, X_plot, output_dir)
            if shap_explanation is not None:
                plot_shap_waterfall(shap_explanation, output_dir)

            # SHAP vs LR comparison only if LightGBM won
            if is_lgbm:
                plot_shap_vs_lr(
                    mean_abs_shap, feature_names, lr_model_path, output_dir
                )

    # ------------------------------------------------------------------
    # 6b. Threshold optimization & business analysis
    # ------------------------------------------------------------------
    # Always compute test-set curve for the diagnostic plot
    _, thresholds, precisions_arr, recalls_arr, f1s_arr = find_optimal_thresholds(y_test, y_prob)

    # Load thresholds from model payload (selected on internal val set — no test leakage).
    # Fall back to test-set computation only for backward compatibility with old payloads.
    if "thresholds" in payload:
        thresholds_dict = payload["thresholds"]
        logger.info("Thresholds loaded from model payload (internal validation — no test-set leakage)")
    else:
        logger.warning("No pre-computed thresholds in payload — using test-set thresholds (diagnostic only)")
        thresholds_dict, _, _, _, _ = find_optimal_thresholds(y_test, y_prob)
    logger.info(f"Best F1 threshold: {thresholds_dict['best_f1_threshold']:.3f}")
    logger.info(f"Business threshold: {thresholds_dict['business_threshold']:.3f} (Precision>=0.60)")

    # Diagnostic threshold plot — computed on test set, labeled accordingly
    plot_threshold_curve(thresholds, precisions_arr, recalls_arr, f1s_arr, thresholds_dict, output_dir)

    # --- Business threshold confusion matrix ---
    y_pred_business = (y_prob >= thresholds_dict["business_threshold"]).astype(int)
    cm_biz = plot_confusion_matrix(
        y_test, y_pred_business, model_name, output_dir,
        filename="confusion_matrix_business_threshold.png",
        subtitle=f"business threshold = {thresholds_dict['business_threshold']:.2f}",
    )
    tn_b, fp_b, fn_b, tp_b = cm_biz.ravel()
    precision_business = safe_div(tp_b, tp_b + fp_b)
    recall_business    = safe_div(tp_b, tp_b + fn_b)
    f1_business        = safe_div(2 * precision_business * recall_business,
                                   precision_business + recall_business)
    logger.info(
        f"Business threshold ({thresholds_dict['business_threshold']:.2f}) — "
        f"Precision: {precision_business:.4f}, Recall: {recall_business:.4f}, F1: {f1_business:.4f}"
    )

    # --- Business segmentation ---
    segments = plot_business_segmentation(y_test, y_prob, output_dir)
    seg_path = os.path.join(output_dir, 'business_segmentation.json')
    with open(seg_path, 'w') as f:
        json.dump(segments, f, indent=2)

    # --- Export predictions ---
    export_predictions(y_test, y_prob, thresholds_dict, output_dir)

    # --- Top SHAP drivers (y_prob_sample is aligned to the SHAP sample) ---
    if mean_abs_shap is not None and shap_values is not None:
        export_top_shap_drivers(shap_values, feature_names, y_prob_sample, output_dir)

    # ------------------------------------------------------------------
    # 7. Save JSON
    # ------------------------------------------------------------------
    results = {
        "model": model_name,
        "n_test_samples": int(len(y_test)),
        "threshold_used": 0.5,
        "metrics": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in final_metrics.items()
        },
        "calibration_approx_ece": round(float(approx_ece), 4),
        "calibration_weighted_ece": round(weighted_ece, 4),
        "calibration_note": (
            "approx ECE > 0.05 — future improvement: CalibratedClassifierCV "
            "on validation set (not test set)"
            if approx_ece > 0.05 else "Calibration OK (approx ECE <= 0.05)"
        ),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn_count), "tp": int(tp),
        },
        "business_interpretation_at_threshold_0_5": {
            "churners_caught_tp": int(tp),
            "churners_missed_fn": int(fn_count),
            "false_alarms_fp": int(fp),
            "catch_rate_recall_pct": round(catch_rate * 100, 1),
            "precision_pct": round(precision_val * 100, 1),
        },
        "threshold_optimization": thresholds_dict,
        "business_segmentation": segments,
        "confusion_matrix_business_threshold": {
            "tn": int(tn_b), "fp": int(fp_b), "fn": int(fn_b), "tp": int(tp_b),
        },
        "business_threshold_metrics": {
            "threshold": thresholds_dict["business_threshold"],
            "precision_business": round(precision_business, 4),
            "recall_business": round(recall_business, 4),
            "f1_business": round(f1_business, 4),
        },
    }

    json_path = os.path.join(output_dir, "final_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to: {json_path}")

    # ------------------------------------------------------------------
    # 8. MLflow artifact logging (non-blocking)
    # ------------------------------------------------------------------
    try:
        import mlflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        mlflow.set_experiment('churn_prediction')
        with mlflow.start_run(run_name='Final_Evaluation'):
            for _fname in [
                'final_evaluation.json', 'model_selection_report.json',
                'business_segmentation.json', 'predictions.csv',
            ]:
                _fpath = os.path.join(output_dir, _fname)
                if os.path.exists(_fpath):
                    try:
                        mlflow.log_artifact(_fpath)
                    except Exception:
                        pass
            for _fname in [
                'roc_curve.png', 'precision_recall_curve.png',
                'calibration_curve.png', 'confusion_matrix.png',
                'confusion_matrix_business_threshold.png', 'gain_curve.png',
                'shap_summary.png', 'shap_bar.png', 'shap_waterfall.png',
                'shap_vs_lr.png', 'precision_recall_vs_threshold.png',
            ]:
                _fpath = os.path.join(output_dir, _fname)
                if os.path.exists(_fpath):
                    try:
                        mlflow.log_artifact(_fpath)
                    except Exception:
                        pass
            logger.info('MLflow artifact logging completed for Final_Evaluation')
    except Exception as _mlflow_e:
        logger.warning(f'MLflow artifact logging failed (non-blocking): {_mlflow_e}')

    logger.info("=" * 60)
    logger.info("TASK 4 completed successfully")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 60)
