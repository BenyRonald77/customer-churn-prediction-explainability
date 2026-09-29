"""Train and compare churn models across imbalance strategies.

Models: Logistic Regression (baseline), Random Forest, XGBoost.
Imbalance strategies per model: none, SMOTE, class_weight balanced.

Metrics on the held-out test set: precision, recall, F1, ROC-AUC, PR-AUC.
Accuracy is reported for reference only, never for model selection.

Run:  .venv/bin/python src/train.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
from xgboost import XGBClassifier

from preprocess import prepare_data, apply_smote, RANDOM_STATE

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
MODELS_DIR = BASE / "models"


def get_models(pos_weight: float):
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced",
                                              random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     random_state=RANDOM_STATE),
        "rf_balanced": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE),
        "xgb": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=RANDOM_STATE, n_jobs=-1),
        "xgb_balanced": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                      subsample=0.8, colsample_bytree=0.8,
                                      scale_pos_weight=pos_weight,
                                      random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, feature_names, pre = prepare_data()
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    X_sm, y_sm = apply_smote(X_train, y_train)

    datasets = {"none": (X_train, y_train), "smote": (X_sm, y_sm)}
    rows = []
    for name, model in get_models(pos_weight).items():
        strategy = "class_weight" if name.endswith("_balanced") else "none"
        # smote variants are trained separately below
        Xd, yd = datasets["none"]
        model.fit(Xd, yd)
        m = evaluate(model, X_test, y_test)
        rows.append({"model": name, "imbalance": strategy, **m})
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        print(f"{name:15s} ({strategy:12s}) " +
              " ".join(f"{k}={v}" for k, v in m.items()))

    # SMOTE variants: same models, no class weights, trained on resampled data
    for name in ("logreg", "rf", "xgb"):
        model = get_models(pos_weight)[name]
        Xd, yd = datasets["smote"]
        model.fit(Xd, yd)
        m = evaluate(model, X_test, y_test)
        rows.append({"model": name, "imbalance": "smote", **m})
        joblib.dump(model, MODELS_DIR / f"{name}_smote.joblib")
        print(f"{name:15s} ({'smote':12s}) " +
              " ".join(f"{k}={v}" for k, v in m.items()))

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    df.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(f"\nsaved -> {REPORTS / 'model_comparison.csv'}")
    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
