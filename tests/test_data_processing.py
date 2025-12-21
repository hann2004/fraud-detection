from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"Processed dataset missing in CI: {path}. Skipping data file validation.")
    return pd.read_csv(path)


def test_ecommerce_processed_schema_and_balance():
    df = _load_csv("ecommerce_train_processed.csv")

    required = {
        "purchase_value",
        "age",
        "time_since_signup_hours",
        "time_of_day",
        "device_usage_count",
        "country_fraud_rate",
        "purchase_value_log",
        "purchase_value_sqrt",
        "purchase_value_category_medium",
        "age_group_middle_aged",
        "unusual_hour_purchase_1",
        "country_freq_encoded",
        "class",
    }

    assert required.issubset(df.columns), "Missing engineered features in e-commerce train set"
    assert not df.isnull().any().any(), "E-commerce processed data should not contain nulls"
    assert set(df["class"].unique()) <= {0, 1}, "Target must be binary"

    ratio = (df["class"] == 0).sum() / (df["class"] == 1).sum()
    assert 1.8 <= ratio <= 2.2, "SMOTE balance should target ~2:1 ratio"

    for col in ["purchase_value", "age", "time_since_signup_hours"]:
        mean = df[col].mean()
        assert abs(mean) < 0.25, f"{col} should be near zero after scaling"


def test_creditcard_processed_schema_and_balance():
    df = _load_csv("creditcard_train_processed.csv")

    required = {
        "Time_hours",
        "Time_sin",
        "Time_cos",
        "Time_diff",
        "Amount_log",
        "Amount_sqrt",
        "Amount_zscore",
        "Amount_rolling_mean",
        "Amount_rolling_std",
        "Amount_high_relative",
        "V14_squared",
        "V10_squared",
        "Amount_V14_interaction",
        "Amount_V10_interaction",
        "combined_anomaly_score",
        "Amount_category_Low",
        "Amount_category_Medium",
        "Amount_category_Extreme",
        "Class",
    }

    assert required.issubset(df.columns), "Missing engineered features in credit card train set"
    assert not df.isnull().any().any(), "Credit card processed data should not contain nulls"
    assert set(df["Class"].unique()) <= {0, 1}, "Target must be binary"

    ratio = (df["Class"] == 0).sum() / (df["Class"] == 1).sum()
    assert 90 <= ratio <= 110, "SMOTE balance should target ~100:1 ratio"

    for col in ["Amount_log", "Amount_sqrt"]:
        mean = df[col].mean()
        assert abs(mean) < 0.1, f"{col} should be centered after scaling"
