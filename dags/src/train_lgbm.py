"""
Task 2b — LightGBM Training
============================
Loads dataset_lgbm.pkl, performs train/test split (same seed as LR
for fair comparison), and trains LightGBM with:

    - Native categorical feature handling (no one-hot encoding needed)
    - Native missing value handling (sparsity-aware splits)
    - Optuna hyperparameter optimisation (train set only — no test leakage)
    - 5-fold stratified cross-validation using best params from Optuna
    - Threshold selection using best params on internal validation split
    - Bootstrap confidence intervals on all metrics

Coherent flow:
    1. train/test split
    2. Optuna optimisation on TRAIN ONLY
    3. retrieve best_params (or fallback to BASELINE_PARAMS)
    4. 5-fold CV using best_params
    5. threshold selection using predictions generated with best_params
    6. final training using best_params on full train set
    7. final evaluation on untouched test set

Why LightGBM over XGBoost:
    - Handles categorical features natively via optimal split finding
      (rather than requiring one-hot encoding)
    - Handles missing values natively like XGBoost
    - Histogram-based algorithm: faster on large datasets (100k rows)
    - was_missing flags kept: distinguish "informative missing" from
      "missing handled in split" — often improves performance
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42       # same as LR — guarantees identical split
TEST_SIZE = 0.20
CV_FOLDS = 5
N_BOOTSTRAP = 300
CONFIDENCE = 0.95
N_OPTUNA_TRIALS = 20

BASELINE_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_metric(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_prob: np.ndarray,
                     metric_fn,
                     n_iter: int = N_BOOTSTRAP,
                     confidence: float = CONFIDENCE) -> tuple:
    """
    Bootstrap confidence interval for a given metric.
    Percentile method (Efron, 1979) — no normality assumption.
    Returns: (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(RANDOM_STATE)
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
            scores.append(score)
        except Exception:
            pass
    if len(scores) == 0:
        return (
            metric_fn(y_true, y_pred, y_prob),
            0.0,
            1.0,
        )
    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    return (
        float(np.mean(scores)),
        float(np.percentile(scores, alpha * 100)),
        float(np.percentile(scores, (1 - alpha) * 100)),
    )


def log_cv_results(cv_results: dict, model_name: str) -> None:
    """Logs cross-validation results with 95% confidence intervals."""
    logger.info(f"--- {model_name} Cross-Validation Results ---")
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores = cv_results[f"test_{metric}"]
        mean = np.mean(scores)
        std = np.std(scores)
        ci_low = mean - 2 * std
        ci_high = mean + 2 * std
        logger.info(
            f"  {metric.upper():12s}: {mean:.4f} +/- {std:.4f}  "
            f"[95% CI: {ci_low:.4f} - {ci_high:.4f}]"
        )
    train_auc = np.mean(cv_results["train_roc_auc"])
    test_auc = np.mean(cv_results["test_roc_auc"])
    gap = train_auc - test_auc
    logger.info(
        f"  Overfitting gap (Train-Test AUC): {gap:.4f} "
        f"({'slight overfitting' if gap > 0.05 else 'OK'})"
    )


def log_feature_importance(model: lgb.LGBMClassifier,
                            feature_names: list) -> None:
    """
    Logs LightGBM feature importance (gain-based).
    Gain = total reduction in loss brought by a feature across all splits.
    More meaningful than split count for comparing features of different
    cardinality.
    """
    importances = model.feature_importances_
    top_n = 20
    top_idx = np.argsort(importances)[::-1][:top_n]
    logger.info("--- Top 20 Features by Gain Importance ---")
    logger.info("  (Gain = total loss reduction — more robust than split count)")
    for rank, idx in enumerate(top_idx, 1):
        logger.info(
            f"  {rank:2d}. {feature_names[idx]:45s}: "
            f"gain = {importances[idx]:,.1f}"
        )


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimisation
# ---------------------------------------------------------------------------

def optimize_hyperparameters(X_train, y_train, n_trials=N_OPTUNA_TRIALS, cat_cols=None):
    """
    3-fold stratified CV on X_train/y_train only — no test-set data used.
    Logs each trial as a nested MLflow run under 'LightGBM_Optuna_Study'.
    Returns (best_params dict, best_cv_auc float).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    _mlflow_active = [False]  # mutable flag visible to the objective closure

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }
        aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr = np.array(y_train)[train_idx]
            y_vl = np.array(y_train)[val_idx]
            m = lgb.LGBMClassifier(
                **params,
                n_estimators=300,
                class_weight="balanced",
                importance_type="gain",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            fit_kwargs = {"categorical_feature": cat_cols} if cat_cols else {}
            m.fit(X_tr, y_tr, **fit_kwargs)
            aucs.append(roc_auc_score(y_vl, m.predict_proba(X_vl)[:, 1]))
        mean_auc = float(np.mean(aucs))
        if _mlflow_active[0]:
            try:
                import mlflow
                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    mlflow.log_param("trial_number", trial.number)
                    for k, v in params.items():
                        mlflow.log_param(k, v)
                    mlflow.log_metric("mean_cv_auc", mean_auc)
            except Exception:
                pass
        return mean_auc

    study = optuna.create_study(direction="maximize")
    try:
        import mlflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        mlflow.set_experiment('churn_prediction')
        with mlflow.start_run(run_name='LightGBM_Optuna_Study'):
            _mlflow_active[0] = True
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_metric("best_cv_auc", study.best_value)
            for k, v in study.best_params.items():
                mlflow.log_param(f"best_{k}", v)
    except Exception as e:
        logger.warning(f"MLflow Optuna run failed (non-blocking): {e}")
        _mlflow_active[0] = False
        if not study.trials:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Optuna best CV AUC: {study.best_value:.4f}")
    logger.info(f"Optuna best params: {study.best_params}")
    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_lgbm(prepared_path: str, model_path: str) -> None:
    """
    Full LightGBM training pipeline.

    Args:
        prepared_path: path to dataset_lgbm.pkl (from preparation task)
        model_path:    path where model pickle will be saved
    """
    logger.info("=" * 60)
    logger.info("TASK 2b: LightGBM Training")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load prepared data
    # ------------------------------------------------------------------
    logger.info(f"Loading data from: {prepared_path}")
    with open(prepared_path, "rb") as f:
        payload = pickle.load(f)

    X = payload["X"]
    y = payload["y"]
    cat_cols = payload["categorical_cols"]
    feature_names = payload["feature_names"]

    logger.info(f"Loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    logger.info(f"Categorical features (native): {cat_cols}")
    logger.info(f"Missing values: {X.isnull().sum().sum():,} (handled natively by LightGBM)")

    # ------------------------------------------------------------------
    # 2. Train/test split — same seed as LR for fair comparison
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,   # identical to LR split
        stratify=y,
    )
    logger.info(
        f"Split: train={len(X_train):,} | test={len(X_test):,} "
        f"(seed={RANDOM_STATE} — same as LR for fair comparison)"
    )

    # ------------------------------------------------------------------
    # 3. Identify categorical feature indices for LightGBM
    # ------------------------------------------------------------------
    cat_cols_present = [c for c in cat_cols if c in X_train.columns]
    cat_feature_indices = [
        list(X_train.columns).index(c)
        for c in cat_cols_present
    ]
    logger.info(
        f"Passing {len(cat_feature_indices)} categorical features "
        f"natively to LightGBM"
    )

    # ------------------------------------------------------------------
    # 4. Optuna hyperparameter search on TRAIN SET ONLY (no test leakage)
    # ------------------------------------------------------------------
    best_params = None
    tuning_method = "fixed_fallback"
    try:
        logger.info(f"Starting Optuna search ({N_OPTUNA_TRIALS} trials, 3-fold CV on train set)...")
        best_params, best_cv_auc = optimize_hyperparameters(X_train, y_train, cat_cols=cat_cols_present)
        tuning_method = "Optuna"
        logger.info(f"Optuna finished — best CV AUC: {best_cv_auc:.4f}")
    except Exception as e:
        logger.warning(f"Optuna failed — falling back to fixed hyperparameters: {e}")

    final_params = best_params if best_params is not None else BASELINE_PARAMS
    logger.info(f"Using params [{tuning_method}]: {final_params}")

    # ------------------------------------------------------------------
    # 5. 5-fold CV using final_params (Optuna best or baseline fallback)
    # ------------------------------------------------------------------
    logger.info(f"Running {CV_FOLDS}-fold stratified CV with {tuning_method} params...")
    logger.info("(LightGBM handles missing values and categoricals natively in each fold)")

    model = lgb.LGBMClassifier(
        **final_params,
        n_estimators=500,
        class_weight="balanced",
        importance_type="gain",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("clf", model)])

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        n_jobs=1,
        return_train_score=True,
    )
    log_cv_results(cv_results, "LightGBM")

    # ------------------------------------------------------------------
    # 5b. Threshold selection via out-of-fold (OOF) predictions on X_train
    #     Covers every training row exactly once — more robust than a single
    #     internal validation split.  Falls back to that split if OOF fails.
    # ------------------------------------------------------------------
    logger.info(f"Computing optimal thresholds via {CV_FOLDS}-fold OOF predictions on X_train...")
    thresholds_payload = None
    try:
        _oof_cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        _oof_labels  = np.array(y_train)
        _X_train_arr = X_train.reset_index(drop=True)
        _oof_probs   = np.zeros(len(_oof_labels))
        for _tr_idx, _val_idx in _oof_cv.split(_X_train_arr, _oof_labels):
            _fold_model = lgb.LGBMClassifier(
                **final_params,
                n_estimators=500,
                class_weight="balanced",
                importance_type="gain",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            _fold_model.fit(
                _X_train_arr.iloc[_tr_idx], _oof_labels[_tr_idx],
                categorical_feature=cat_cols_present,
            )
            _oof_probs[_val_idx] = _fold_model.predict_proba(_X_train_arr.iloc[_val_idx])[:, 1]
            del _fold_model
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
            f"Thresholds (OOF {CV_FOLDS}-fold, {tuning_method}): "
            f"best_f1={thresholds_payload['best_f1_threshold']:.3f}, "
            f"business={thresholds_payload['business_threshold']:.3f}"
        )
    except Exception as _oof_e:
        logger.warning(f"OOF threshold selection failed ({_oof_e}) — falling back to internal validation split")
    if thresholds_payload is None:
        X_train_inner, X_val, y_train_inner, y_val = train_test_split(
            X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_train,
        )
        _temp_model = lgb.LGBMClassifier(
            **final_params,
            n_estimators=500,
            class_weight="balanced",
            importance_type="gain",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        _temp_model.fit(X_train_inner, y_train_inner, categorical_feature=cat_cols_present)
        _val_prob     = _temp_model.predict_proba(X_val)[:, 1]
        del _temp_model
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
            f"Thresholds (fallback internal val, {tuning_method}): "
            f"best_f1={thresholds_payload['best_f1_threshold']:.3f}, "
            f"business={thresholds_payload['business_threshold']:.3f}"
        )

    # ------------------------------------------------------------------
    # 6. Final training on full training set using final_params
    # ------------------------------------------------------------------
    logger.info(f"Training final model on full training set using {tuning_method} params...")
    model.fit(X_train, y_train, categorical_feature=cat_cols_present)
    logger.info("Training completed!")

    # ------------------------------------------------------------------
    # 7. Feature importance (gain-based)
    # ------------------------------------------------------------------
    log_feature_importance(model, feature_names)

    # ------------------------------------------------------------------
    # 8. Hold-out evaluation with bootstrap confidence intervals
    # ------------------------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    logger.info("--- Hold-out Test Set Metrics (Bootstrap CI 95%) ---")
    metrics = {
        "AUC-ROC":   lambda yt, yp, ypr: roc_auc_score(yt, ypr),
        "F1":        lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
        "Precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
        "Recall":    lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        "Avg Prec":  lambda yt, yp, ypr: average_precision_score(yt, ypr),
    }
    holdout_metrics = {}
    for name, fn in metrics.items():
        mean, lo, hi = bootstrap_metric(
            np.array(y_test), np.array(y_pred), np.array(y_prob), fn
        )
        holdout_metrics[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        logger.info(
            f"  {name:12s}: {mean:.4f}  "
            f"[95% CI: {lo:.4f} - {hi:.4f}]"
        )

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_payload = {
        "model": model,
        "model_name": "LightGBM",
        "feature_names": feature_names,
        "cat_cols": cat_cols_present,
        "cat_feature_indices": cat_feature_indices,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "holdout_metrics": holdout_metrics,
        "cv_results": cv_results,
        "thresholds": thresholds_payload,
        "best_params": final_params,
        "tuning_method": tuning_method,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)

    # ------------------------------------------------------------------
    # 10. MLflow logging — log actual params used (not hardcoded baseline)
    # ------------------------------------------------------------------
    try:
        import mlflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        mlflow.set_experiment('churn_prediction')
        with mlflow.start_run(run_name='LightGBM'):
            log_params = {
                "n_estimators": 500,
                "tuning_method": tuning_method,
                **final_params,
                "threshold_0_5": thresholds_payload["threshold_0_5"],
                "best_f1_threshold": thresholds_payload["best_f1_threshold"],
                "business_threshold": thresholds_payload["business_threshold"],
                "threshold_selection_source": thresholds_payload["selection_source"],
                "business_rule": thresholds_payload["business_rule"],
                "n_categorical_features": len(cat_cols_present),
                "n_features": len(feature_names),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
            }
            mlflow.log_params(log_params)
            for metric_name, metric_vals in holdout_metrics.items():
                mlflow.log_metric(
                    metric_name.replace('-', '_').replace(' ', '_'),
                    metric_vals['mean'],
                )
            try:
                mlflow.lightgbm.log_model(model, 'lgbm_model')
                mlflow.set_tag('model_artifact_logged', 'true')
            except Exception as artifact_e:
                mlflow.set_tag('model_artifact_logged', 'false')
                logger.warning(f'MLflow model logging failed, continuing because pickle model was saved: {artifact_e}')
            logger.info('MLflow logging completed for LightGBM')
    except Exception as e:
        logger.warning(f'MLflow logging failed (non-blocking): {e}')

    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)
    logger.info("TASK 2b completed successfully")
    logger.info("=" * 60)
