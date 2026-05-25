import numpy as np
import pandas as pd
import os
import json
import pickle
import logging
import mlflow

from typing import Optional, Any

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from classification_evaluator import ClassificationEvaluator


# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# MLflow configuration
mlflow.set_tracking_uri(
    "http://ec2-16-16-216-150.eu-north-1.compute.amazonaws.com:5000"
)

mlflow.set_experiment(
    "ride-driver-acceptance"
)


# Load MLflow run ID
def load_run_id() -> Optional[str]:

    try:

        logger.info(
            "Loading MLflow run ID..."
        )

        run_info_path = os.path.join(
            "models",
            "run_info.json"
        )

        with open(run_info_path, "r") as f:

            run_info = json.load(f)
            run_id = run_info["run_id"]

        logger.info(
            "Run ID loaded successfully."
        )

        return run_id

    except Exception:

        logger.exception(
            "Error loading run ID."
        )

    return None


# Load model and scaler
def load_model_and_scaler(
    model_path: str,
    scaler_path: str
) -> tuple[
    Optional[Any],
    Optional[StandardScaler]
]:

    try:

        logger.info(
            "Loading model and scaler..."
        )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        logger.info(
            "Model and scaler loaded successfully."
        )

        return model, scaler

    except Exception:

        logger.exception(
            "Unexpected error while loading model/scaler."
        )

    return None, None


# Load validation data
def load_validation_data(
    val_path: str
) -> Optional[pd.DataFrame]:

    try:

        logger.info(
            "Loading validation data..."
        )

        val_df = pd.read_csv(val_path)

        logger.info(
            "Validation data loaded successfully."
        )

        return val_df

    except Exception:

        logger.exception(
            "Unexpected error while loading validation data."
        )

    return None


# Prepare validation data
def prepare_validation_data(
    val_df: pd.DataFrame,
    scaler: StandardScaler
) -> tuple[
    Optional[np.ndarray],
    Optional[pd.Series]
]:

    try:

        logger.info(
            "Preparing validation data..."
        )

        X_val = val_df.drop(
            columns=['target']
        )

        y_val = val_df[
            'target'
        ]

        X_val = scaler.transform(
            X_val
        )

        logger.info(
            "Validation data preparation complete."
        )

        return X_val, y_val

    except Exception:

        logger.exception(
            "Unexpected error during validation preparation."
        )

    return None, None


# Save evaluation report
def save_evaluation_report(
    evaluator: ClassificationEvaluator,
    y_val: pd.Series
) -> None:

    try:

        logger.info(
            "Saving evaluation report..."
        )

        report = {
            "accuracy": accuracy_score(
                y_val,
                evaluator.y_pred
            ),
            "precision": precision_score(
                y_val,
                evaluator.y_pred
            ),
            "recall": recall_score(
                y_val,
                evaluator.y_pred
            ),
            "f1_score": f1_score(
                y_val,
                evaluator.y_pred
            ),
            "roc_auc": roc_auc_score(
                y_val,
                evaluator.y_prob
            )
        }

        os.makedirs(
            "reports",
            exist_ok=True
        )

        output_path = os.path.join(
            "reports",
            "evaluation_report.json"
        )

        with open(output_path, "w") as f:

            json.dump(
                report,
                f,
                indent=4
            )

        logger.info(
            "Evaluation report saved successfully."
        )

    except Exception:

        logger.exception(
            "Unexpected error while saving evaluation report."
        )


def main() -> None:

    try:

        run_id = load_run_id()

        if run_id is None:

            logger.error(
                "Run ID loading failed."
            )

            return

        model, scaler = load_model_and_scaler(
            "models/lightgbm_model.pkl",
            "models/scaler.pkl"
        )

        if model is None or scaler is None:

            logger.error(
                "Model/scaler loading failed."
            )

            return

        val_df = load_validation_data(
            "data/processed/test.csv"
        )

        if val_df is None:

            logger.error(
                "Validation data loading failed."
            )

            return

        X_val, y_val = prepare_validation_data(
            val_df,
            scaler
        )

        if X_val is None or y_val is None:

            logger.error(
                "Validation data preparation failed."
            )

            return

        with mlflow.start_run(
            run_id=run_id
        ):

            logger.info(
                "Initializing evaluator..."
            )

            evaluator = ClassificationEvaluator(
                model,
                X_val,
                y_val
            )

            logger.info(
                "Generating evaluation metrics..."
            )

            evaluator.basic_metrics()

            evaluator.classification_report()

            evaluator.confusion_matrix_plot()

            evaluator.threshold_analysis()

            accuracy = accuracy_score(
                y_val,
                evaluator.y_pred
            )

            precision = precision_score(
                y_val,
                evaluator.y_pred
            )

            recall = recall_score(
                y_val,
                evaluator.y_pred
            )

            f1 = f1_score(
                y_val,
                evaluator.y_pred
            )

            roc_auc = roc_auc_score(
                y_val,
                evaluator.y_prob
            )

            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "precision",
                precision
            )

            mlflow.log_metric(
                "recall",
                recall
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

            mlflow.log_metric(
                "roc_auc",
                roc_auc
            )

            save_evaluation_report(
                evaluator,
                y_val
            )

            mlflow.log_artifact(
                "reports/evaluation_report.json"
            )

            logger.info(
                "Model evaluation pipeline completed successfully."
            )

    except Exception:

        logger.exception(
            "Unexpected error in evaluation pipeline."
        )


if __name__ == "__main__":
    main()