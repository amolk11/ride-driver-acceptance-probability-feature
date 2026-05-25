import json

import numpy as np
import pandas as pd
import os
import yaml
import pickle
import logging
import mlflow
import mlflow.lightgbm

from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation
)

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


# Load parameters
def load_params(param_path: str) -> Optional[dict]:

    try:
        logger.info(
            f"Loading parameters from {param_path}..."
        )

        with open(param_path, "r") as f:
            params = yaml.safe_load(f)["model_building"]

        logger.info("Parameters loaded successfully.")

        return params

    except FileNotFoundError:
        logger.exception("Parameter file not found.")

    except yaml.YAMLError:
        logger.exception("Error parsing YAML file.")

    except KeyError:
        logger.exception(
            "'model_building' section missing in YAML."
        )

    except Exception:
        logger.exception(
            "Unexpected error while loading parameters."
        )

    return None


# Load data
def load_data(train_path: str) -> pd.DataFrame:

    try:
        logger.info(
            f"Loading training data from {train_path} "
            )

        train_df = pd.read_csv(train_path)
        logger.info("Data loaded successfully.")

        return train_df

    except FileNotFoundError:
        logger.exception("Data file not found.")

    except pd.errors.EmptyDataError:
        logger.exception("CSV file is empty.")

    except pd.errors.ParserError:
        logger.exception("Error parsing CSV file.")

    except Exception:
        logger.exception(
            "Unexpected error while loading data."
        )

    return None, None


# Split data into features and target
def split_data(train_df: pd.DataFrame) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[pd.Series]
]:

    try:
        logger.info(
            "Splitting data into features and target..."
        )

        if 'target' not in train_df.columns:
            raise KeyError(
                "'target' column missing."
            )

        X = train_df.drop(columns=['target'])
        y = train_df['target']

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        logger.info("Data splitting complete.")

        return X_train, X_val, y_train, y_val

    except KeyError:
        logger.exception(
            "Target column missing during split."
        )

    except ValueError:
        logger.exception(
            "Error during train-test split."
        )

    except Exception:
        logger.exception(
            "Unexpected error during data split."
        )

    return None, None, None, None


# Standardize features
def standardize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Standardizing features...")

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val),columns=X_val.columns)

        logger.info(
            "Feature standardization complete."
        )

        logger.info("Saving scaler...")

        scaler_path = os.path.join(
            "models",
            "scaler.pkl"
        )

        os.makedirs("models", exist_ok=True)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info("Scaler saved successfully.")

        return X_train_scaled, X_val_scaled

    except PermissionError:
        logger.exception(
            "Permission denied while saving scaler."
        )

    except Exception:
        logger.exception(
            "Unexpected error during standardization."
        )

    return None, None


# Train model
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict
) -> Optional[LGBMClassifier]:

    try:
        logger.info("Training LightGBM model...")

        required_params = [
            'learning_rate',
            'n_estimators',
            'num_leaves',
            'min_child_samples',
            'subsample',
            'colsample_bytree',
            'reg_alpha',
            'reg_lambda',
            'class_weight',
            'random_state'
        ]

        missing_params = [
            param for param in required_params
            if param not in params
        ]

        if missing_params:
            raise KeyError(
                f"Missing parameters: {missing_params}"
            )

        model = LGBMClassifier(
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            num_leaves=params['num_leaves'],
            min_child_samples=params['min_child_samples'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            class_weight=params['class_weight'],
            random_state=params['random_state']
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)
            ]
        )

        logger.info("Model training complete.")

        return model

    except KeyError:
        logger.exception(
            "Required model parameters missing."
        )

    except ValueError:
        logger.exception(
            "Invalid values passed to model."
        )

    except Exception:
        logger.exception(
            "Unexpected error during model training."
        )

    return None


# Save model
def save_model(model: LGBMClassifier) -> None:

    try:
        logger.info("Saving model...")

        os.makedirs("models", exist_ok=True)

        model_path = os.path.join(
            "models",
            "lightgbm_model.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved successfully.")

    except PermissionError:
        logger.exception(
            "Permission denied while saving model."
        )

    except Exception:
        logger.exception(
            "Unexpected error while saving model."
        )


def save_run_info(run_id: str) -> None:
    try:
        logger.info("Saving run info...")

        run_info_path = os.path.join(
            "models",
            f"run_info.json"
        )

        with open(run_info_path, "w") as f:
            json.dump({"run_id": run_id}, f, indent=4)

        logger.info("Run info saved successfully.")
        
    except Exception:
        logger.exception("Unexpected error while saving run info.")


def main() -> None:

    try:

        params = load_params("params.yaml")

        if params is None:
            logger.error(
                "Parameter loading failed."
            )
            return

        train_df = load_data(
            "data/processed/train.csv"
        )

        if train_df is None:
            logger.error(
                "Data loading failed."
            )
            return

        X_train, X_val, y_train, y_val = split_data(
            train_df
        )

        if any(
            x is None
            for x in [
                X_train,
                X_val,
                y_train,
                y_val
            ]
        ):
            logger.error(
                "Data splitting failed."
            )
            return

        X_train_scaled, X_val_scaled = (
            standardize_features(
                X_train,
                X_val
            )
        )

        if (
            X_train_scaled is None
            or X_val_scaled is None
        ):
            logger.error(
                "Feature standardization failed."
            )
            return

        with mlflow.start_run() as run:

            mlflow.log_params(params)

            model = train_model(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                params
            )

            if model is None:
                logger.error(
                    "Model training failed."
                )
                return

            y_pred = model.predict(
                X_val_scaled
            )

            y_proba = model.predict_proba(
                X_val_scaled
            )[:, 1]

            accuracy = accuracy_score(
                y_val,
                y_pred
            )

            precision = precision_score(
                y_val,
                y_pred
            )

            recall = recall_score(
                y_val,
                y_pred
            )

            f1 = f1_score(
                y_val,
                y_pred
            )

            roc_auc = roc_auc_score(
                y_val,
                y_proba
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

            mlflow.lightgbm.log_model(
                model,
                artifact_path="model"
            )

            save_model(model)

            mlflow.log_artifact(
                "models/scaler.pkl"
            )

            mlflow.log_artifact(
                "models/lightgbm_model.pkl"
            )
            
            save_run_info(run.info.run_id)

            logger.info(
                "MLflow tracking completed successfully."
            )

    except Exception:
        logger.exception(
            "Unexpected error in model building pipeline."
        )


if __name__ == "__main__":
    main()