"""
EDA Pipeline — DAG
===================
Exploratory Data Analysis pipeline — run once manually before
the main churn pipeline to understand the data.

DAG 1 of 2. See churn_dag.py for the main ML pipeline.

Schedule: None (manual trigger only — run once)
"""

import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

BASE_DIR   = "/opt/airflow"
DATA_PATH  = os.path.join(BASE_DIR, "data",    "dataset.csv")
EDA_OUTPUT = os.path.join(BASE_DIR, "outputs", "eda")


def run_eda(**context):
    import logging
    from src.eda import run_eda
    logging.basicConfig(level=logging.INFO)
    run_eda(
        data_path=DATA_PATH,
        output_dir=EDA_OUTPUT,
    )


with DAG(
    dag_id="eda_pipeline",
    description="Exploratory Data Analysis — run once before churn_prediction_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,       # manual only — run once
    catchup=False,
    tags=["eda", "exploration", "once"],
    doc_md="""
## EDA Pipeline

Standalone exploratory analysis — run this ONCE before the main pipeline.

### Outputs (saved to outputs/eda/)
1. `01_target_distribution.png`   — churn vs no-churn counts and proportions
2. `02_missing_values.png`        — missing rate per column with thresholds
3. `03_numeric_distributions.png` — top 12 features split by churn
4. `04_outlier_detection.png`     — Z-score outlier percentage per feature
5. `05_churn_by_categorical.png`  — churn rate per category value
6. `06_correlation_heatmap.png`   — feature correlation matrix
7. `07_top_features_boxplot.png`  — boxplots with Mann-Whitney U test
8. `eda_summary.json`             — summary statistics
    """,
) as dag:

    task_eda = PythonOperator(
        task_id="exploratory_analysis",
        python_callable=run_eda,
    )
