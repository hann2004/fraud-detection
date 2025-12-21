"""Data preparation utilities for fraud datasets.

This module centralizes the feature engineering and transformation steps that
were previously implemented in notebooks. Functions are intentionally
lightweight and composable so they can be reused by scripts or tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SplitArtifacts:
    """Container for train/test splits and the fitted scaler."""

    train: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def _categorize_hour(hour: int) -> str:
    if 0 <= hour < 6:
        return "night"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening"


def _time_of_day_code(label: str) -> int:
    mapping = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
    return mapping.get(label, -1)


def load_ecommerce_cleaned(path: Path | str) -> pd.DataFrame:
    """Load cleaned e-commerce data with parsed timestamps."""

    df = pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])
    logger.info("Loaded e-commerce cleaned data: %s rows", len(df))
    return df


def engineer_ecommerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create e-commerce fraud features.

    This mirrors the notebook feature engineering: temporal, behavioral,
    country risk, and value-based signals plus one-hot encodings.
    """

    data = df.copy()
    data["purchase_time"] = pd.to_datetime(data["purchase_time"])
    data["signup_time"] = pd.to_datetime(data["signup_time"])

    data["purchase_month"] = data["purchase_time"].dt.month
    data["purchase_day_of_month"] = data["purchase_time"].dt.day
    data["purchase_minute"] = data["purchase_time"].dt.minute
    data["signup_hour"] = data["signup_time"].dt.hour
    data["signup_day"] = data["signup_time"].dt.dayofweek

    data["time_since_signup_days"] = data["time_since_signup_hours"] / 24
    data["time_since_signup_weeks"] = data["time_since_signup_days"] / 7
    data["time_of_day"] = data["purchase_hour"].apply(_categorize_hour)
    data["is_weekend"] = (data["purchase_day"] >= 5).astype(int)

    data = data.sort_values(["user_id", "purchase_time"])
    data["time_since_last_txn_hours"] = (
        data.groupby("user_id")["purchase_time"].diff().dt.total_seconds() / 3600
    )
    data["time_since_last_txn_hours"] = data["time_since_last_txn_hours"].fillna(30 * 24)

    data["device_usage_count"] = data["device_id"].map(data["device_id"].value_counts())

    country_fraud_rate = data.groupby("country")["class"].mean()
    data["country_fraud_rate"] = data["country"].map(country_fraud_rate).fillna(0)
    risk_bins = [0, 0.01, 0.03, np.inf]
    risk_labels = ["low_risk", "medium_risk", "high_risk"]
    data["country_risk_category"] = pd.cut(
        data["country_fraud_rate"], bins=risk_bins, labels=risk_labels, include_lowest=True
    )

    data["purchase_value_log"] = np.log1p(data["purchase_value"])
    data["purchase_value_sqrt"] = np.sqrt(data["purchase_value"])

    pv_bins = [-np.inf, 50, 150, np.inf]
    pv_labels = ["low", "medium", "very_high"]
    data["purchase_value_category"] = pd.cut(
        data["purchase_value"], bins=pv_bins, labels=pv_labels, include_lowest=True
    )

    age_bins = [0, 25, 45, np.inf]
    age_labels = ["young", "middle_aged", "senior"]
    data["age_group"] = pd.cut(data["age"], bins=age_bins, labels=age_labels, include_lowest=True)

    data["is_new_user"] = (data["time_since_signup_hours"] <= 24).astype(int)
    data["high_value_new_user"] = (
        (data["is_new_user"] == 1) & (data["purchase_value"] >= 200)
    ).astype(int)
    data["unusual_hour_purchase"] = data["purchase_hour"].isin([0, 1, 2, 3, 4]).astype(int)

    country_freq = data["country"].value_counts(normalize=True)
    data["country_freq_encoded"] = data["country"].map(country_freq)

    data["time_of_day"] = data["time_of_day"].map(_time_of_day_code).astype(int)

    drop_cols = [
        "signup_time",
        "purchase_time",
        "device_id",
        "ip_address",
        "ip_address_int",
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    ]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    categorical_cols = [
        "source",
        "browser",
        "sex",
        "purchase_day",
        "purchase_month",
        "signup_day",
        "is_weekend",
        "country_risk_category",
        "purchase_value_category",
        "age_group",
        "is_new_user",
        "high_value_new_user",
        "unusual_hour_purchase",
    ]

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    data = data.fillna(0)
    logger.info("Engineered e-commerce features: %s columns", len(data.columns))
    return data


def transform_ecommerce(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    smote_strategy: float = 0.5,
) -> SplitArtifacts:
    """Scale, split, and balance the e-commerce dataset."""

    if "class" not in data.columns:
        raise ValueError("Expected column 'class' in e-commerce dataset")

    features = data.drop(columns=["class"])
    target = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=random_state
    )

    binary_cols = [c for c in features.columns if set(features[c].dropna().unique()) <= {0, 1}]
    numeric_cols = [c for c in features.columns if c not in binary_cols]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if numeric_cols:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    else:
        logger.warning("No numeric columns to scale for e-commerce dataset")

    try:
        sampler = SMOTE(random_state=random_state, sampling_strategy=smote_strategy)
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
    except Exception:  # pragma: no cover - fallback path
        sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=smote_strategy)
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)

    train_df = pd.concat([X_res, y_res.rename("class")], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.rename("class")], axis=1)
    return SplitArtifacts(train=train_df, test=test_df, scaler=scaler)


def load_creditcard_cleaned(path: Path | str) -> pd.DataFrame:
    """Load cleaned credit card data."""

    df = pd.read_csv(path)
    logger.info("Loaded credit card cleaned data: %s rows", len(df))
    return df


def engineer_creditcard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature set for the credit card dataset."""

    data = df.copy()
    data = data.sort_values("Time")

    data["Time_hours"] = data["Time"] / 3600
    data["Time_sin"] = np.sin(2 * np.pi * data["Time_hours"] / 24)
    data["Time_cos"] = np.cos(2 * np.pi * data["Time_hours"] / 24)
    data["Time_diff"] = data["Time"].diff().fillna(0)

    data["Amount_log"] = np.log1p(data["Amount"])
    data["Amount_sqrt"] = np.sqrt(data["Amount"])
    amount_mean = data["Amount"].mean()
    amount_std = data["Amount"].std(ddof=0) or 1.0
    data["Amount_zscore"] = (data["Amount"] - amount_mean) / amount_std

    window = 100
    data["Amount_rolling_mean"] = data["Amount"].rolling(window, min_periods=1).mean()
    data["Amount_rolling_std"] = data["Amount"].rolling(window, min_periods=1).std().fillna(0)
    data["Amount_high_relative"] = (
        data["Amount"] > (data["Amount_rolling_mean"] + 3 * data["Amount_rolling_std"])
    ).astype(int)

    data["V14_squared"] = data["V14"] ** 2
    data["V10_squared"] = data["V10"] ** 2
    data["Amount_V14_interaction"] = data["Amount"] * data["V14"]
    data["Amount_V10_interaction"] = data["Amount"] * data["V10"]

    for col in ["V14", "V10", "V12", "V16", "V17"]:
        std = data[col].std(ddof=0) or 1.0
        data[f"{col}_zscore"] = (data[col] - data[col].mean()) / std

    anomaly_cols = ["V14_zscore", "V10_zscore", "V12_zscore", "V16_zscore", "V17_zscore", "Amount_zscore"]
    data["combined_anomaly_score"] = data[anomaly_cols].abs().sum(axis=1)

    amount_bins = [-np.inf, 10, 50, 150, 500, 1000, 2000, np.inf]
    amount_labels = [
        "Amount_category_Low",
        "Amount_category_Medium",
        "Amount_category_High",
        "Amount_category_Very High",
        "Amount_category_Premium",
        "Amount_category_Luxury",
        "Amount_category_Extreme",
    ]
    data["Amount_category"] = pd.cut(data["Amount"], bins=amount_bins, labels=amount_labels)
    data = pd.get_dummies(data, columns=["Amount_category"], drop_first=False)

    data = data.fillna(0)
    logger.info("Engineered credit card features: %s columns", len(data.columns))
    return data


def transform_creditcard(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    smote_strategy: float = 0.01,
) -> SplitArtifacts:
    """Scale non-PCA features, stratify split, and rebalance."""

    if "Class" not in data.columns:
        raise ValueError("Expected column 'Class' in credit card dataset")

    features = data.drop(columns=["Class"])
    target = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=random_state
    )

    new_features = [col for col in features.columns if not col.startswith("V") and col not in {"Time", "Amount"}]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if new_features:
        X_train_scaled[new_features] = scaler.fit_transform(X_train[new_features])
        X_test_scaled[new_features] = scaler.transform(X_test[new_features])
    else:
        logger.warning("No additional features to scale for credit card dataset")

    try:
        sampler = SMOTE(random_state=random_state, sampling_strategy=smote_strategy)
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
    except Exception:  # pragma: no cover - fallback path
        sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=smote_strategy)
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)

    train_df = pd.concat([X_res, y_res.rename("Class")], axis=1)
    test_df = pd.concat([X_test_scaled, y_test.rename("Class")], axis=1)
    return SplitArtifacts(train=train_df, test=test_df, scaler=scaler)


def save_processed(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path | str, *, train_name: str, test_name: str) -> Tuple[Path, Path]:
    """Persist processed splits to CSV."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / train_name
    test_path = output / test_name
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logger.info("Saved processed datasets: %s, %s", train_path, test_path)
    return train_path, test_path


__all__ = [
    "SplitArtifacts",
    "load_ecommerce_cleaned",
    "engineer_ecommerce_features",
    "transform_ecommerce",
    "load_creditcard_cleaned",
    "engineer_creditcard_features",
    "transform_creditcard",
    "save_processed",
]
