"""
Churn Prediction Pipeline — Main DAG
=====================================
Implements the full ML lifecycle for telecom churn prediction.

Architecture:
    data_preparation
           |
    +------+------+
    |             |
  train_lr    train_lgbm    <- parallel execution
    |             |
    +------+------+
           |
    model_selection
           |
    final_evaluation

DAG 2 of 2. See eda_dag.py for the exploratory analysis pipeline.
"""

import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Paths — all relative to /opt/airflow (Docker mount point)
# ---------------------------------------------------------------------------
BASE_DIR          = "/opt/airflow"
DATA_PATH         = os.path.join(BASE_DIR, "data",    "dataset.csv")
PREPARED_LR_PATH  = os.path.join(BASE_DIR, "models",  "dataset_lr_raw.pkl")
PREPARED_LGBM_PATH= os.path.join(BASE_DIR, "models",  "dataset_lgbm.pkl")
LR_MODEL_PATH     = os.path.join(BASE_DIR, "models",  "lr_model.pkl")
LGBM_MODEL_PATH   = os.path.join(BASE_DIR, "models",  "lgbm_model.pkl")
BEST_MODEL_PATH   = os.path.join(BASE_DIR, "models",  "best_model.pkl")
OUTPUT_DIR        = os.path.join(BASE_DIR, "outputs")


# ---------------------------------------------------------------------------
# Task wrappers
# ---------------------------------------------------------------------------

def run_preparation(**context):
    import logging
    from src.preparation import load_and_prepare
    logging.basicConfig(level=logging.INFO)
    load_and_prepare(
        data_path=DATA_PATH,
        output_lr_path=PREPARED_LR_PATH,
        output_lgbm_path=PREPARED_LGBM_PATH,
    )


def run_train_lr(**context):
    import logging
    from src.train_lr import train_logistic_regression
    logging.basicConfig(level=logging.INFO)
    train_logistic_regression(
        prepared_path=PREPARED_LR_PATH,
        model_path=LR_MODEL_PATH,
    )


def run_train_lgbm(**context):
    import logging
    from src.train_lgbm import train_lgbm
    logging.basicConfig(level=logging.INFO)
    train_lgbm(
        prepared_path=PREPARED_LGBM_PATH,
        model_path=LGBM_MODEL_PATH,
    )


def run_model_selection(**context):
    import logging
    from src.model_selection import select_model
    logging.basicConfig(level=logging.INFO)
    select_model(
        lr_model_path=LR_MODEL_PATH,
        lgbm_model_path=LGBM_MODEL_PATH,
        best_model_path=BEST_MODEL_PATH,
        output_dir=OUTPUT_DIR,
    )


def run_evaluation(**context):
    import logging
    from src.evaluation import evaluate_model
    logging.basicConfig(level=logging.INFO)
    evaluate_model(
        best_model_path=BEST_MODEL_PATH,
        lr_model_path=LR_MODEL_PATH,
        output_dir=OUTPUT_DIR,
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="churn_prediction_pipeline",
    description=(
        "Telecom churn prediction: "
        "preparation → [LR || LightGBM] → selection → evaluation + SHAP"
    ),
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["churn", "ml", "shap", "telecom", "production"],
    doc_md="""
## Churn Prediction Pipeline

Full ML lifecycle for telecom customer churn prediction.

### Steps
1. **data_preparation** — loads dataset.csv, fixes formats, creates
   was_missing flags, engineers 3 domain features, exports two
   optimised datasets (LR raw + LightGBM with native categoricals).

2. **train_logistic_regression** *(parallel)* — leakage-free pipeline:
   ColumnTransformer(TargetEncoder) → SimpleImputer → RobustScaler → LR(L2).
   5-fold CV, bootstrap CI, standardised coefficients.

3. **train_lightgbm** *(parallel)* — native categoricals and missing values,
   gain-based importance, 5-fold CV, bootstrap CI.

4. **model_selection** — bootstrap AUC difference + McNemar test.
   Prefers simpler model (LR) if no significant difference.
   Considers recall for business relevance.

5. **final_evaluation** — full metrics + calibration curve + SHAP
   (TreeExplainer for LightGBM, LinearExplainer for LR) + SHAP vs LR
   coefficient comparison. All plots saved to outputs/.

### Outputs
- `models/dataset_lr_raw.pkl`   — prepared dataset for LR
- `models/dataset_lgbm.pkl`     — prepared dataset for LightGBM
- `models/lr_model.pkl`         — trained LR pipeline
- `models/lgbm_model.pkl`       — trained LightGBM model
- `models/best_model.pkl`       — winner model
- `outputs/*.png`               — all evaluation and SHAP plots
- `outputs/model_selection_report.json`
- `outputs/final_evaluation.json`
    """,
) as dag:

    task_preparation = PythonOperator(
        task_id="data_preparation",
        python_callable=run_preparation,
    )

    task_train_lr = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=run_train_lr,
    )

    task_train_lgbm = PythonOperator(
        task_id="train_lightgbm",
        python_callable=run_train_lgbm,
    )

    task_model_selection = PythonOperator(
        task_id="model_selection",
        python_callable=run_model_selection,
    )

    task_evaluation = PythonOperator(
        task_id="final_evaluation",
        python_callable=run_evaluation,
    )

    # Execution order:
    # preparation → [train_lr || train_lgbm] → model_selection → evaluation
    task_preparation >> [task_train_lr, task_train_lgbm]
    [task_train_lr, task_train_lgbm] >> task_model_selection
    task_model_selection >> task_evaluation
