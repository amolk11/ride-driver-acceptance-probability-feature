import numpy as np
import pandas as pd
import os
import yaml
import logging

from typing import Optional
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from missing_value_imputer import BookingValueImputer

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

# load params
def load_params(file_path: str) -> dict:
    try:
        logger.info("Loading parameters...")

        with open(file_path, 'r') as f:
            params = yaml.safe_load(f)['feature_engineering']
            
        logger.info("Parameters loaded successfully.")

        return params

    except FileNotFoundError:
        logger.exception("Parameter file not found.")

    except ValueError:
        logger.exception("Error parsing parameter file.")

    except Exception:
        logger.exception("Unexpected error while loading parameters.")

    return None

# Load data
def load_data(file_path: str) -> pd.DataFrame:

    try:
        logger.info("Loading data...")

        df = pd.read_csv(file_path)

        logger.info("Data loaded successfully.")

        return df

    except FileNotFoundError:
        logger.exception("File not found.")

    except pd.errors.EmptyDataError:
        logger.exception("CSV file is empty.")

    except pd.errors.ParserError:
        logger.exception("Error parsing CSV file.")

    except Exception:
        logger.exception("Unexpected error while loading data.")

    return None


# Convert Datetime
def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:

    try:
        logger.info("Converting Datetime column...")

        df['Datetime'] = pd.to_datetime(df['Datetime'],errors='coerce')

        logger.info("Datetime conversion complete.")

        return df

    except KeyError:
        logger.exception("Datetime column missing.")

    except Exception:
        logger.exception("Unexpected error during datetime conversion.")

    return None


# Time features
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:

    try:
        logger.info("Creating time features...")

        df['day_of_week'] = df['Datetime'].dt.dayofweek

        df['month'] = df['Datetime'].dt.month

        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        df["is_peak_hour"] = (df['Datetime'].dt.hour.isin([7, 8, 9, 17, 18, 19])).astype(int)


        df.drop(columns=['Datetime'], inplace=True)

        logger.info("Time features created successfully.")

        return df

    except KeyError:
        logger.exception("Datetime column missing.")

    except Exception:
        logger.exception("Unexpected error during time feature creation.")

    return None, None


# Encode categorical features
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    try:
        logger.info("Encoding categorical features...")

        vehicle_price_order = [[
            'eBike',
            'Bike',
            'Auto',
            'Go Mini',
            'Go Sedan',
            'Premier Sedan',
            'Uber XL'
        ]]

        encoder = OrdinalEncoder(categories=vehicle_price_order)

        df[['Vehicle Type']] = encoder.fit_transform(df[['Vehicle Type']])

        logger.info("Categorical encoding complete.")

        return df

    except KeyError:
        logger.exception("Vehicle Type column missing.")

    except ValueError:
        logger.exception("Unexpected category found.")

    except Exception:
        logger.exception("Unexpected error during encoding.")

    return None


# split data
def split_data_into_train_test_data(df: pd.DataFrame, params: dict) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:
    
    train_df, test_df = train_test_split(df, test_size=params['test_size'], random_state=params['random_state'], shuffle=True, stratify=df['target'])
    
    return train_df, test_df


# Handle missing values
def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[
    Optional[pd.DataFrame], 
    Optional[pd.DataFrame]
]:
    
    try:
        logger.info("Handling missing values...")

        imputer = BookingValueImputer()

        train_df = imputer.fit_transform(train_df)
        test_df = imputer.transform(test_df)

        logger.info("Missing values handled successfully.")

        return train_df, test_df

    except KeyError:
        logger.exception("Required columns missing for imputation.")

    return None, None


# Create location features
def create_location_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating location features...")

        pickup_freq = train_df['Pickup Location'].value_counts()

        train_df['pickup_freq'] = (
            train_df['Pickup Location'].map(pickup_freq)
        )

        test_df['pickup_freq'] = (
            test_df['Pickup Location'].map(pickup_freq).fillna(0)
        )

        drop_freq = train_df['Drop Location'].value_counts()

        train_df['drop_freq'] = (
            train_df['Drop Location'].map(drop_freq)
        )

        test_df['drop_freq'] = (
            test_df['Drop Location'].map(drop_freq).fillna(0)
        )

        logger.info("Location features created successfully.")

        return train_df, test_df

    except KeyError:
        logger.exception("Pickup/Drop Location columns missing.")

    except Exception:
        logger.exception("Unexpected error during location feature creation.")

    return None, None


# Route feature
def create_route_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating route features...")

        train_df['route'] = (
            train_df['Pickup Location']
            + "_"
            + train_df['Drop Location']
        )

        test_df['route'] = (
            test_df['Pickup Location']
            + "_"
            + test_df['Drop Location']
        )

        route_avg = (
            train_df.groupby('route')['Booking Value'].mean()
        )

        train_df['route_avg_price'] = (
            train_df['route'].map(route_avg)
        )

        test_df['route_avg_price'] = (
            test_df['route'].map(route_avg).fillna(
                train_df['Booking Value'].mean()
            )
        )

        train_df['price_vs_route'] = (
            train_df['Booking Value']
            / train_df['route_avg_price']
        )

        test_df['price_vs_route'] = (
            test_df['Booking Value']
            / test_df['route_avg_price']
        )

        train_df.drop(
            columns=['route', 'Pickup Location', 'Drop Location'],
            inplace=True
        )

        test_df.drop(
            columns=['route', 'Pickup Location', 'Drop Location'],
            inplace=True
        )

        logger.info("Route features created successfully.")

        return train_df, test_df

    except KeyError:
        logger.exception("Required columns missing for route features.")

    except ZeroDivisionError:
        logger.exception("Division by zero encountered.")

    except Exception:
        logger.exception("Unexpected error during route feature creation.")

    return None, None


# Feature transformation
def feature_transformation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Applying feature transformation...")

        train_df['log_price'] = np.log1p(train_df['Booking Value'])
        test_df['log_price'] = np.log1p(test_df['Booking Value'])

        train_df['log_route_avg'] = np.log1p(
            train_df['route_avg_price']
        )

        test_df['log_route_avg'] = np.log1p(
            test_df['route_avg_price']
        )

        train_df['is_high_value'] = (
            train_df['price_vs_route'] > 1
        ).astype(int)

        test_df['is_high_value'] = (
            test_df['price_vs_route'] > 1
        ).astype(int)

        logger.info("Feature transformation complete.")

        return train_df, test_df

    except Exception:
        logger.exception("Unexpected error during feature transformation.")

    return None, None


# Location popularity features
def create_location_popularity_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Creating location popularity features...")

        train_df['location_demand'] = (
            train_df['pickup_freq']
            + train_df['drop_freq']
        )

        train_df['location_interaction'] = (
            train_df['pickup_freq']
            * train_df['drop_freq']
        )

        test_df['location_demand'] = (
            test_df['pickup_freq']
            + test_df['drop_freq']
        )

        test_df['location_interaction'] = (
            test_df['pickup_freq']
            * test_df['drop_freq']
        )

        train_df.drop(
            columns=['pickup_freq', 'drop_freq'],
            inplace=True
        )

        test_df.drop(
            columns=['pickup_freq', 'drop_freq'],
            inplace=True
        )

        logger.info("Location popularity features created.")

        return train_df, test_df

    except Exception:
        logger.exception(
            "Unexpected error during location popularity feature creation."
        )

    return None, None


# Feature selection
def select_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:

    try:
        logger.info("Selecting features...")

        drop_cols = [
            'Booking Value',
            'route_avg_price',
            'location_interaction',
            'month',
            'Booking Status'
        ]

        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

        logger.info("Feature selection complete.")

        return train_df, test_df

    except KeyError:
        logger.exception("Columns missing during feature selection.")

    except Exception:
        logger.exception("Unexpected error during feature selection.")

    return None, None


# Save engineered data
def save_engineered_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:

    try:
        logger.info("Saving engineered data...")

        data_path = os.path.join("data", "processed")

        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(
            os.path.join(data_path, "train.csv"),
            index=False
        )

        test_df.to_csv(
            os.path.join(data_path, "test.csv"),
            index=False
        )

        logger.info("Engineered data saved successfully.")

    except PermissionError:
        logger.exception("Permission denied while saving data.")

    except Exception:
        logger.exception("Unexpected error while saving data.")


def main() -> None:

    try:

        df = load_data("data/raw/driver_acceptance_data.csv")

        if df is None:
            return

        df = convert_datetime(df)

        df = create_time_features(df)

        df = encode_categorical_features(df)
        
        params = load_params("params.yaml")

        train_df, test_df = split_data_into_train_test_data(df, params)
        
        train_df, test_df = handle_missing_values(train_df, test_df)

        train_df, test_df = create_location_features(train_df, test_df)

        train_df, test_df = create_route_feature(train_df, test_df)

        train_df, test_df = feature_transformation(train_df, test_df)

        train_df, test_df = create_location_popularity_features(train_df, test_df)

        train_df, test_df = select_features(train_df, test_df)

        save_engineered_data(train_df, test_df)

    except Exception:
        logger.exception("Unexpected error in feature engineering pipeline.")


if __name__ == "__main__":
    main()