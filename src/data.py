"""Dataset loading utilities for model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURES_DATA_FILE

TARGET_COL = "higher"
ENCODED_TARGET_COL = "target_risk"
DROP_COLS = [TARGET_COL, ENCODED_TARGET_COL, "G3"]
RANDOM_STATE = 42


def load_feature_dataset() -> pd.DataFrame:
    """Load the processed feature dataset."""

    if not FEATURES_DATA_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found: {FEATURES_DATA_FILE}")

    return pd.read_csv(FEATURES_DATA_FILE)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Divide two series while keeping zero denominators stable."""

    return np.where(denominator == 0, 0, numerator / denominator)


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the feature engineering used to train the final model."""

    df_fe = df.copy()

    if {"G1", "G2"}.issubset(df_fe.columns):
        df_fe["grade_avg_12"] = df_fe[["G1", "G2"]].mean(axis=1)
        df_fe["grade_trend_12"] = df_fe["G2"] - df_fe["G1"]
        df_fe["grade_momentum"] = safe_divide(df_fe["G2"] - df_fe["G1"], df_fe["G1"] + 1)

    if "absences" in df_fe.columns:
        df_fe["high_absence_flag"] = (
            df_fe["absences"] >= df_fe["absences"].quantile(0.90)
        ).astype(int)

    if {"Dalc", "Walc"}.issubset(df_fe.columns):
        df_fe["alc_index"] = (df_fe["Dalc"] + df_fe["Walc"]) / 2

    if {"goout", "freetime"}.issubset(df_fe.columns):
        df_fe["social_index"] = (df_fe["goout"] + df_fe["freetime"]) / 2

    if {"alc_index", "social_index"}.issubset(df_fe.columns):
        df_fe["risk_behavior_index"] = (
            df_fe["alc_index"] + df_fe["social_index"]
        ) / 2

    binary_cols = [
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "internet",
        "romantic",
    ]
    for column in binary_cols:
        if column in df_fe.columns:
            df_fe[column] = df_fe[column].map({"yes": 1, "no": 0})

    support_components = [
        column
        for column in ["schoolsup", "famsup", "paid", "activities", "internet", "nursery"]
        if column in df_fe.columns
    ]
    if support_components:
        df_fe["support_index"] = df_fe[support_components].fillna(0).sum(axis=1)

    if {"Medu", "Fedu"}.issubset(df_fe.columns):
        df_fe["parent_edu_avg"] = df_fe[["Medu", "Fedu"]].mean(axis=1)
        df_fe["parent_edu_gap"] = (df_fe["Medu"] - df_fe["Fedu"]).abs()

    if "failures" in df_fe.columns:
        df_fe["failure_flag"] = (df_fe["failures"] > 0).astype(int)
        df_fe["high_failure_flag"] = (df_fe["failures"] >= 2).astype(int)

    return df_fe


def load_dataset_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the train/test split used in the notebook and final evaluation."""

    df = add_model_features(load_feature_dataset())
    df[ENCODED_TARGET_COL] = df[TARGET_COL].map({"yes": 0, "no": 1}).astype("int64")

    y = df[ENCODED_TARGET_COL]
    X = df.drop(columns=DROP_COLS, errors="ignore")

    if "absences" in X.columns:
        cap = X["absences"].quantile(0.99)
        X["absences"] = np.minimum(X["absences"], cap)

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
