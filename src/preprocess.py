"""Preprocessing + imbalance handling for the Telco churn dataset.

Steps:
  1. Drop customerID (identifier, not predictive).
  2. Target Churn: Yes/No -> 1/0.
  3. Normalize service flags: "No phone service"/"No internet service" -> "No".
  4. ColumnTransformer: numeric -> median imputation + scaling;
     categorical -> one-hot encoding.
  5. Stratified train/test split (80/20, seed 42).
  6. Imbalance strategies prepared for comparison in train.py:
     - none (baseline)
     - SMOTE oversampling on the training split
     - class_weight="balanced" (applied at model level)

Run:  .venv/bin/python src/preprocess.py
"""
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_raw, TARGET

BASE = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
MODELS_DIR = BASE / "models"
RANDOM_STATE = 42

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
SERVICE_FLAGS = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                 "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies"]
OTHER_CAT = ["gender", "Partner", "Dependents", "Contract", "PaperlessBilling",
             "PaymentMethod"]
DROP_COLS = ["customerID"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning before the ColumnTransformer."""
    df = df.copy()
    df = df.drop(columns=DROP_COLS)
    # Service flags: "No phone service"/"No internet service" mean "No"
    for col in SERVICE_FLAGS:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No phone service": "No", "No internet service": "No"})
    df["target"] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=[TARGET])


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, SERVICE_FLAGS + OTHER_CAT),
    ])


def prepare_data():
    """Return (X_train, X_test, y_train, y_test, feature_names, preprocessor)."""
    df = clean(load_raw())
    X = df.drop(columns=["target"])
    y = df["target"].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pre = build_preprocessor()
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)
    feature_names = list(pre.get_feature_names_out())
    return X_train, X_test, y_train, y_test, feature_names, pre


def apply_smote(X_train, y_train):
    """SMOTE oversampling on the (already preprocessed) training split."""
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X_train, y_train)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, feature_names, pre = prepare_data()
    print(f"train: {X_train.shape}, test: {X_test.shape}, features: {len(feature_names)}")
    print(f"train class balance: 0={(y_train == 0).sum()} 1={(y_train == 1).sum()}")

    X_res, y_res = apply_smote(X_train, y_train)
    print(f"after SMOTE: {X_res.shape}, 0={(y_res == 0).sum()} 1={(y_res == 1).sum()}")

    pd.DataFrame(X_train, columns=feature_names).to_parquet(
        PROCESSED_DIR / "X_train.parquet", index=False)
    pd.DataFrame(X_test, columns=feature_names).to_parquet(
        PROCESSED_DIR / "X_test.parquet", index=False)
    pd.DataFrame({"target": y_train}).to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    pd.DataFrame({"target": y_test}).to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    print(f"saved processed data -> {PROCESSED_DIR}, preprocessor -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
