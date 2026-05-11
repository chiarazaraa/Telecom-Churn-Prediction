"""
Task 2a — Logistic Regression Training
=======================================
Loads dataset_lr_raw.pkl, performs train/test split, then applies
all statistically-sensitive operations AFTER the split to avoid leakage.

Pipeline (leakage-free):
    ColumnTransformer(TargetEncoder) -> SimpleImputer -> RobustScaler -> LR

Key design decisions:
    - TargetEncoder is INSIDE the Pipeline so it is refit on each CV
      fold's training data only. No leakage into validation folds.
    - sklearn.set_config(transform_output="pandas") preserves feature
      names through the pipeline transformations.
    - Multicollinearity drop computed on X_train only (before pipeline).
    - RobustScaler: uses median/IQR — robust to outliers found in EDA.
    - L2 regularisation (C=0.1): strong regularisation for many features.

Mathematical notes:
    - Logistic Regression: log(p/(1-p)) = X*beta
      Coefficients represent change in log-odds per unit feature increase.
    - After RobustScaler, features are on comparable scale ->
      coefficients are directly comparable across features.
    - Target Encoding with empirical Bayes smoothing:
      encoded = (n*cat_mean + m*global_mean) / (n+m)
      Handles rare/unseen categories by shrinking toward global churn rate.
    - Bootstrap CI (Efron 1979): percentile method, no normality assumption.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)

# Preserve feature names through pipeline transformations
sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)

RANDOM_STATE           = 42
TEST_SIZE              = 0.20
CV_FOLDS               = 5
CORRELATION_THRESHOLD  = 0.85
N_BOOTSTRAP            = 300
CONFIDENCE             = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_multicollinear_features(X_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  num_cols: list,
                                  y_train: pd.Series) -> tuple:
    """
    Drops highly correlated numeric features (Pearson > threshold).
    Computed ONLY on X_train to avoid leakage.
    When two features correlate above threshold, keeps the one with
    higher absolute Pearson correlation with the target.
    """
    logger.info(f"Checking multicollinearity (threshold: {CORRELATION_THRESHOLD})...")
    num_train    = X_train[num_cols].copy()
    corr_matrix  = num_train.corr().abs()
    target_corr  = num_train.corrwith(y_train.astype(float)).abs()

    dropped = set()
    cols    = list(num_cols)

    for i in range(len(cols)):
        if cols[i] in dropped:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in dropped:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > CORRELATION_THRESHOLD:
                keep = (cols[i]
                        if target_corr.get(cols[i], 0) >= target_corr.get(cols[j], 0)
                        else cols[j])
                drop = cols[j] if keep == cols[i] else cols[i]
                dropped.add(drop)
                logger.info(
                    f"  Dropping '{drop}' (corr={corr_matrix.loc[cols[i], cols[j]]:.3f}"
                    f" with '{keep}', target_corr kept={target_corr.get(keep,0):.3f})"
                )

    dropped_list = list(dropped)
    X_train = X_train.drop(columns=dropped_list, errors="ignore")
    X_test  = X_test.drop(columns=dropped_list,  errors="ignore")
    logger.info(f"Dropped {len(dropped_list)} multicollinear features")
    return X_train, X_test, dropped_list


def bootstrap_metric(y_true, y_pred, y_prob, metric_fn,
                     n_iter=N_BOOTSTRAP, confidence=CONFIDENCE):
    """Bootstrap CI via percentile method (Efron, 1979)."""
    rng    = np.random.RandomState(RANDOM_STATE)
    scores = []
    n      = len(y_true)
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
    alpha  = (1 - confidence) / 2
    return (float(np.mean(scores)),
            float(np.percentile(scores, alpha * 100)),
            float(np.percentile(scores, (1 - alpha) * 100)))


def log_cv_results(cv_results: dict) -> None:
    logger.info("--- Logistic Regression Cross-Validation Results ---")
    # F1/Precision/Recall use default threshold 0.5; AUC is threshold-free.
    # Threshold optimisation is performed separately on an internal validation split afterward.
    logger.info("  (F1/P/R at threshold 0.5 | AUC threshold-free | threshold tuning done post-CV)")
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores   = cv_results[f"test_{metric}"]
        mean     = np.mean(scores)
        std      = np.std(scores)
        ci_low   = mean - 2 * std
        ci_high  = mean + 2 * std
        logger.info(
            f"  {metric.upper():12s}: {mean:.4f} +/- {std:.4f}  "
            f"[95% CI: {ci_low:.4f} - {ci_high:.4f}]"
        )
    gap = np.mean(cv_results["train_roc_auc"]) - np.mean(cv_results["test_roc_auc"])
    logger.info(
        f"  Overfitting gap: {gap:.4f} "
        f"({'slight overfitting' if gap > 0.05 else 'OK'})"
    )


def log_coefficients(pipeline: Pipeline, feature_names: list) -> None:
    """Logs top standardised LR coefficients with business direction."""
    clf   = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    if len(coefs) != len(feature_names):
        logger.warning(
            f"Coefficient length ({len(coefs)}) != "
            f"feature_names ({len(feature_names)}) — skipping coefficient log"
        )
        return
    top_n   = 20
    top_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    logger.info("--- Top 20 LR Coefficients (after RobustScaler) ---")
    logger.info("  Positive beta -> increases churn log-odds | Negative -> decreases")
    for rank, idx in enumerate(top_idx, 1):
        direction = "↑ increases" if coefs[idx] > 0 else "↓ decreases"
        logger.info(
            f"  {rank:2d}. {feature_names[idx]:45s}: "
            f"beta = {coefs[idx]:+.4f}  ({direction} churn risk)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_logistic_regression(prepared_path: str, model_path: str) -> None:
    """
    Full LR training pipeline with leakage-free preprocessing.

    Args:
        prepared_path: path to dataset_lr_raw.pkl (from preparation task)
        model_path:    path where model pickle will be saved
    """
    logger.info("=" * 60)
    logger.info("TASK 2a: Logistic Regression Training")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info(f"Loading data from: {prepared_path}")
    with open(prepared_path, "rb") as f:
        payload = pickle.load(f)

    X          = payload["X"]
    y          = payload["y"]
    cat_cols   = payload["categorical_cols"]
    bin_cat_cols = payload["binary_cat_cols"]
    logger.info(f"Loaded: {X.shape[0]:,} samples, {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 2. Train/test split — BEFORE any fitting operation
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Split: train={len(X_train):,} | test={len(X_test):,} (seed={RANDOM_STATE})")

    # ------------------------------------------------------------------
    # 3. Drop multicollinear numeric features (X_train only)
    # ------------------------------------------------------------------
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    num_cols_for_corr = [
        c for c in num_cols
        if not c.endswith("_was_missing") and c not in bin_cat_cols
    ]
    X_train, X_test, dropped_cols = drop_multicollinear_features(
        X_train, X_test, num_cols_for_corr, y_train
    )

    # ------------------------------------------------------------------
    # 4. Prepare categorical column list
    # ------------------------------------------------------------------
    all_cat_cols = [
        c for c in cat_cols + bin_cat_cols
        if c in X_train.columns
    ]
    logger.info(f"Categorical features for TargetEncoder: {len(all_cat_cols)}")

    # Convert categoricals to string for TargetEncoder
    for col in all_cat_cols:
        X_train[col] = X_train[col].astype(str).str.strip().str.upper()
        X_test[col]  = X_test[col].astype(str).str.strip().str.upper()

    # ------------------------------------------------------------------
    # 5. Build leakage-free Pipeline
    #    TargetEncoder is INSIDE the Pipeline ->
    #    refit on each CV fold's training data only
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "target_enc",
                TargetEncoder(
                    smooth="auto",
                    target_type="binary",
                    random_state=RANDOM_STATE,
                ),
                all_cat_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,  # prevents "remainder__" prefix
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # Global median imputation is acceptable: TargetEncoder outputs numeric values
        # before this step, so all columns are numeric at imputation time.
        # Separate numeric/categorical preprocessing branches could be a future improvement.
        ("imputer",       SimpleImputer(strategy="median")),
        ("scaler",        RobustScaler()),
        # LR used as interpretable baseline; stronger/nonlinear models expected to outperform.
        # LogisticRegressionCV could be a future improvement for built-in C selection.
        ("clf",           LogisticRegression(
            C=0.1,
            penalty="l2",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            solver="lbfgs",
        )),
    ])

    # ------------------------------------------------------------------
    # 6. Cross-validation — no leakage (TargetEncoder refit per fold)
    # ------------------------------------------------------------------
    logger.info(f"Running {CV_FOLDS}-fold stratified CV (TargetEncoder inside Pipeline)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        n_jobs=-1,
        return_train_score=True,
    )
    log_cv_results(cv_results)

    # ------------------------------------------------------------------
    # 6b. Threshold selection via out-of-fold (OOF) predictions on X_train
    #     Covers every training row exactly once — more robust than a single
    #     internal validation split.  Falls back to that split if OOF fails.
    # ------------------------------------------------------------------
    logger.info(f"Computing optimal thresholds via {CV_FOLDS}-fold OOF predictions on X_train...")
    from sklearn.base import clone as _clone_pipeline
    thresholds_payload = None
    try:
        _oof_cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _oof_labels  = np.array(y_train)
        _X_train_arr = X_train.reset_index(drop=True)
        _oof_probs   = np.zeros(len(_oof_labels))
        for _tr_idx, _val_idx in _oof_cv.split(_X_train_arr, _oof_labels):
            _fp = _clone_pipeline(pipeline)
            _fp.fit(_X_train_arr.iloc[_tr_idx], _oof_labels[_tr_idx])
            _oof_probs[_val_idx] = _fp.predict_proba(_X_train_arr.iloc[_val_idx])[:, 1]
            del _fp
        _thr  = np.linspace(0, 1, 200)
        _prec = np.array([precision_score(_oof_labels, (_oof_probs >= t).astype(int), zero_division=0) for t in _thr])
        _rec  = np.array([recall_score(_oof_labels,    (_oof_probs >= t).astype(int), zero_division=0) for t in _thr])
        _f1   = np.array([f1_score(_oof_labels,        (_oof_probs >= t).astype(int), zero_division=0) for t in _thr])
        _best_f1_idx = int(np.argmax(_f1))
        _biz_mask    = _prec >= 0.60
        _biz_thr     = float(_thr[np.where(_biz_mask)[0][np.argmax(_rec[_biz_mask])]]) if _biz_mask.any() else 0.5
        thresholds_payload = {
            "threshold_0_5":      0.5,
            "best_f1_threshold":  float(_thr[_best_f1_idx]),
            "business_threshold": _biz_thr,
            "selection_source":   "out_of_fold_cross_validation",
            "business_rule":      "max recall with precision >= 0.60",
        }
        logger.info(
            f"Thresholds (OOF {CV_FOLDS}-fold): "
            f"best_f1={thresholds_payload['best_f1_threshold']:.3f}, "
            f"business={thresholds_payload['business_threshold']:.3f}"
        )
    except Exception as _oof_e:
        logger.warning(f"OOF threshold selection failed ({_oof_e}) — falling back to internal validation split")
    if thresholds_payload is None:
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_train,
        )
        _temp_pipeline = _clone_pipeline(pipeline)
        _temp_pipeline.fit(X_train_inner, y_train_inner)
        _val_prob     = _temp_pipeline.predict_proba(X_val)[:, 1]
        del _temp_pipeline
        _thresholds   = np.linspace(0, 1, 200)
        _precisions_t = np.array([precision_score(y_val, (_val_prob >= t).astype(int), zero_division=0) for t in _thresholds])
        _recalls_t    = np.array([recall_score(y_val,    (_val_prob >= t).astype(int), zero_division=0) for t in _thresholds])
        _f1s_t        = np.array([f1_score(y_val,        (_val_prob >= t).astype(int), zero_division=0) for t in _thresholds])
        _best_f1_idx  = int(np.argmax(_f1s_t))
        _biz_mask     = _precisions_t >= 0.60
        _business_thr = float(_thresholds[np.where(_biz_mask)[0][np.argmax(_recalls_t[_biz_mask])]]) if _biz_mask.any() else 0.5
        thresholds_payload = {
            "threshold_0_5":      0.5,
            "best_f1_threshold":  float(_thresholds[_best_f1_idx]),
            "business_threshold": _business_thr,
            "selection_source":   "internal_validation",
            "business_rule":      "max recall with precision >= 0.60",
        }
        logger.info(
            f"Thresholds (fallback internal val): "
            f"best_f1={thresholds_payload['best_f1_threshold']:.3f}, "
            f"business={thresholds_payload['business_threshold']:.3f}"
        )

    # ------------------------------------------------------------------
    # 7. Final training on full training set
    # ------------------------------------------------------------------
    logger.info("Training final model on full training set...")
    pipeline.fit(X_train, y_train)
    logger.info("Training completed!")

    # Extract feature names after pipeline fits
    try:
        feature_names = pipeline[:-1].get_feature_names_out().tolist()
    except Exception:
        feature_names = list(X_train.columns)
    logger.info(f"Final feature count: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 8. Hold-out evaluation with bootstrap CI
    # ------------------------------------------------------------------
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    logger.info("--- Hold-out Test Set Metrics (Bootstrap CI 95%) ---")
    metrics_fns = {
        "AUC-ROC":   lambda yt, yp, ypr: roc_auc_score(yt, ypr),
        "F1":        lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
        "Precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
        "Recall":    lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        "Avg Prec":  lambda yt, yp, ypr: average_precision_score(yt, ypr),
    }
    holdout_metrics = {}
    for name, fn in metrics_fns.items():
        mean, lo, hi = bootstrap_metric(
            np.array(y_test), np.array(y_pred), np.array(y_prob), fn
        )
        holdout_metrics[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        logger.info(f"  {name:12s}: {mean:.4f}  [95% CI: {lo:.4f} - {hi:.4f}]")

    # ------------------------------------------------------------------
    # 9. Log coefficients
    # ------------------------------------------------------------------
    log_coefficients(pipeline, feature_names)

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_payload = {
        "model":           pipeline,
        "model_name":      "Logistic Regression",
        "feature_names":   feature_names,
        "cat_cols":        all_cat_cols,
        "dropped_cols":    dropped_cols,
        "X_test":          X_test,
        "y_test":          y_test,
        "y_prob":          y_prob,
        "y_pred":          y_pred,
        "holdout_metrics": holdout_metrics,
        "cv_results":      cv_results,
        "thresholds":      thresholds_payload,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)

    # --- MLflow logging ---
    try:
        import mlflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        mlflow.set_experiment('churn_prediction')
        with mlflow.start_run(run_name='Logistic_Regression'):
            mlflow.log_params({
                'C': 0.1,
                'penalty': 'l2',
                'max_iter': 1000,
                'scaler': 'RobustScaler',
                'target_encoder': 'sklearn_TargetEncoder_smooth_auto',
                'class_weight': 'balanced',
                'correlation_threshold': CORRELATION_THRESHOLD,
                'n_dropped_multicollinear_features': len(dropped_cols),
                'threshold_0_5': thresholds_payload['threshold_0_5'],
                'best_f1_threshold': thresholds_payload['best_f1_threshold'],
                'business_threshold': thresholds_payload['business_threshold'],
                'threshold_selection_source': thresholds_payload['selection_source'],
                'business_rule': thresholds_payload['business_rule'],
            })
            for metric_name, metric_vals in holdout_metrics.items():
                mlflow.log_metric(metric_name.replace('-', '_').replace(' ', '_'), metric_vals['mean'])
            try:
                mlflow.sklearn.log_model(pipeline, 'lr_model')
                mlflow.set_tag('model_artifact_logged', 'true')
            except Exception as artifact_e:
                mlflow.set_tag('model_artifact_logged', 'false')
                logger.warning(f'MLflow model logging failed, continuing because pickle model was saved: {artifact_e}')
            logger.info('MLflow logging completed for Logistic Regression')
    except Exception as e:
        logger.warning(f'MLflow logging failed (non-blocking): {e}')

    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)
    logger.info("TASK 2a completed successfully")
    logger.info("=" * 60)
