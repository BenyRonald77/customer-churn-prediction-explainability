"""CLI inference for churn prediction.

Loads the saved preprocessor + best model, predicts churn probability for
one customer, and lists the top SHAP factors behind the prediction.

Usage:
  .venv/bin/python predict.py --json '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes",...}'
  .venv/bin/python predict.py --file customer.json [--threshold 0.4]

The JSON must contain the raw feature fields (same names as the raw CSV,
without customerID/Churn). Missing fields are filled with NaN and imputed.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DEFAULT_THRESHOLD = 0.40  # best F1 @recall>=0.75, see reports/threshold_tuning.md

EXPECTED_FIELDS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                   "PhoneService", "MultipleLines", "InternetService",
                   "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                   "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
                   "TotalCharges"]


def load_artifacts(model_path: str | None):
    pre = joblib.load(BASE / "models" / "preprocessor.joblib")
    model_path = model_path or (BASE / "models" / "xgb_balanced.joblib")
    model = joblib.load(model_path)
    return pre, model


def predict_one(payload: dict, threshold: float, model_path: str | None):
    pre, model = load_artifacts(model_path)
    row = {f: payload.get(f, np.nan) for f in EXPECTED_FIELDS}
    X_raw = pd.DataFrame([row])
    # same light cleaning as training
    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        X_raw[col] = X_raw[col].replace(
            {"No phone service": "No", "No internet service": "No"})
    X = pre.transform(X_raw)
    proba = float(model.predict_proba(X)[0, 1])
    label = "CHURN" if proba >= threshold else "TIDAK CHURN"

    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]
    names = list(pre.get_feature_names_out())
    order = np.argsort(-np.abs(sv))[:5]
    factors = []
    for i in order:
        fname = names[i].split("__", 1)[-1].replace("_", " ")
        direction = "mendorong churn" if sv[i] > 0 else "menahan churn"
        factors.append(f"{fname} ({sv[i]:+.3f}): {direction}")
    return proba, label, factors


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict churn for one customer.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="customer features as a JSON string")
    src.add_argument("--file", help="path to a JSON file with customer features")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--model", default=None, help="override model joblib path")
    args = ap.parse_args()

    payload = json.loads(args.json) if args.json else json.load(open(args.file))
    proba, label, factors = predict_one(payload, args.threshold, args.model)
    print(f"Probabilitas churn : {proba:.1%}")
    print(f"Prediksi           : {label} (threshold={args.threshold})")
    print("Faktor utama:")
    for f in factors:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
