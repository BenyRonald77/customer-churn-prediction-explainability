"""Evaluation deep-dive + threshold tuning for the best churn model.

- Picks the best model by PR-AUC from reports/model_comparison.csv.
- Plots precision-recall and ROC curves for the top-3 models.
- Tunes the decision threshold: best F1, and best F1 under recall >= 0.75
  (business-friendly: catching churners matters more than precision here).
- Writes reports/threshold_tuning.md with the numbers.

Run:  .venv/bin/python src/evaluate.py   (after src/train.py)
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             precision_score, recall_score, f1_score)

from preprocess import prepare_data

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_model(name: str):
    return joblib.load(BASE / "models" / f"{name}.joblib")


def main() -> None:
    comp = pd.read_csv(BASE / "reports" / "model_comparison.csv")
    top3 = comp.head(3)
    print("Top-3 by PR-AUC:")
    print(top3[["model", "imbalance", "pr_auc", "f1", "recall"]].to_markdown(index=False))

    _, X_test, _, y_test, _, _ = prepare_data()
    probas = {row["model"]: load_model(row["model"]).predict_proba(X_test)[:, 1]
              for _, row in top3.iterrows()}

    # PR + ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[0].plot(rec, prec, label=f"{name} (AP={auc(rec, prec):.3f})")
    axes[0].set_title("Precision-Recall curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=8)
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title("ROC curve")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "curves_top3.png", dpi=120)
    plt.close()

    # Threshold tuning on the best model
    best = top3.iloc[0]
    p = probas[best["model"]]
    rows = []
    for thr in np.arange(0.1, 0.91, 0.05):
        pred = (p >= thr).astype(int)
        rows.append({
            "threshold": round(float(thr), 2),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
        })
    tdf = pd.DataFrame(rows)
    best_f1 = tdf.loc[tdf["f1"].idxmax()]
    recall_ok = tdf[tdf["recall"] >= 0.75]
    best_rec = recall_ok.loc[recall_ok["f1"].idxmax()] if not recall_ok.empty else None

    lines = ["# Threshold Tuning", "",
             f"Model: `{best['model']}` (imbalance: {best['imbalance']})", "",
             "## Metrics per threshold", "", tdf.to_markdown(index=False), "",
             "## Recommended thresholds",
             f"- Best F1: threshold={best_f1['threshold']:.2f} "
             f"(P={best_f1['precision']}, R={best_f1['recall']}, F1={best_f1['f1']})"]
    if best_rec is not None:
        lines.append(
            f"- Best F1 with recall>=0.75: threshold={best_rec['threshold']:.2f} "
            f"(P={best_rec['precision']}, R={best_rec['recall']}, F1={best_rec['f1']})")
    lines += ["",
              "Note: default 0.5 is rarely optimal on imbalanced data. "
              "Pick the threshold that matches the business cost of missing a churner."]
    (BASE / "reports" / "threshold_tuning.md").write_text("\n".join(lines))
    print(f"\nbest F1 threshold: {best_f1['threshold']:.2f}")
    if best_rec is not None:
        print(f"best F1 @recall>=0.75: {best_rec['threshold']:.2f}")
    print("wrote reports/threshold_tuning.md and reports/figures/curves_top3.png")


if __name__ == "__main__":
    main()
