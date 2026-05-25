import os

import pandas as pd
import logging

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("error.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """

    try:
        logger.info("Loading dataset...")

        df = pd.read_csv(file_path)

        logger.info("Dataset loaded successfully.")
        logger.debug(f"Dataset shape: {df.shape}")

        return df

    except FileNotFoundError:
        logger.exception("File not found.")

    except pd.errors.EmptyDataError:
        logger.exception("CSV file is empty.")

    except pd.errors.ParserError:
        logger.exception("Error parsing CSV file.")

    raise


def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Datetime column to pandas datetime type.
    """

    try:
        logger.info("Converting Datetime column...")

        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

        invalid_dates = df['Datetime'].isna().sum()

        if invalid_dates > 0:
            logger.warning(f"{invalid_dates} invalid datetime values found.")

        logger.info("Datetime conversion complete.")

        return df

    except KeyError:
        logger.exception("Datetime column not found.")
        raise


def create_target_and_select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select required features and create target column.
    """

    try:
        logger.info("Starting feature selection and target creation...")

        features = [
            "Datetime",
            "Vehicle Type",
            "Pickup Location",
            "Drop Location",
            "Booking Value",
        ]

        target_col = "Booking Status"

        required_col = features + [target_col]

        missing_cols = list(set(required_col) - set(df.columns))

        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}")

        df = df[required_col].copy()

        logger.info("Feature selection completed.")

        invalid_statuses = {"Cancelled by Customer","Incomplete"}

        df = df[~df[target_col].isin(invalid_statuses)].copy()

        logger.info("Invalid rows removed.")

        df["target"] = (df[target_col] == "Completed").astype("int8")

        logger.info("Target column created successfully.")

        return df

    except KeyError as e:
        logger.exception(f"Column validation failed: {e}")
        raise

    except Exception as e:
        logger.exception(f"Unexpected error during feature engineering: {e}")
        raise
    
    
def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save preprocessed dataset to CSV file.
    """

    try:
        logger.info("Saving preprocessed dataset...")

        df.to_csv(output_path, index=False)

        logger.info(f"Preprocessed dataset saved to {output_path}.")

    except Exception:
        logger.exception("Error saving preprocessed dataset.")
        raise

def main() -> None:
    
    input_file_path = os.path.join("data", "external", "uber.csv")
    df = load_data(input_file_path)
    df = convert_datetime(df)
    df = create_target_and_select_features(df)
    output_file_path = os.path.join("data", "raw", "driver_acceptance_data.csv")
    save_data(df, output_file_path)

if __name__ == "__main__":
    main()