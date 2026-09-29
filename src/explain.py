"""SHAP explainability for the best churn model.

- Global: mean |SHAP| feature importance (bar plot + CSV).
- Local: waterfall plots for example customers (high-risk churner,
  low-risk customer) plus a human-readable narrative of the top factors.

Run:  .venv/bin/python src/explain.py   (after src/train.py)
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_loader import load_raw
from preprocess import prepare_data, clean, RANDOM_STATE
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = "xgb_balanced"


def pretty_feature(name: str) -> str:
    """Turn 'cat__Contract_Month-to-month' into readable text."""
    short = name.split("__", 1)[-1].replace("_", " ")
    return short


def narrative(customer: pd.Series, shap_vals: np.ndarray,
              feature_names: list[str], proba: float, n: int = 5) -> list[str]:
    order = np.argsort(-np.abs(shap_vals))[:n]
    lines = [f"Probabilitas churn model: {proba:.1%}", ""]
    push_up, push_down = [], []
    for i in order:
        fname = pretty_feature(feature_names[i])
        val = customer.iloc[i]
        sval = shap_vals[i]
        val_txt = f"{val:.2f}" if isinstance(val, (int, float, np.floating)) else str(val)
        entry = f"{fname} = {val_txt} ({sval:+.3f})"
        (push_up if sval > 0 else push_down).append(entry)
    if push_up:
        lines.append("Faktor pendorong churn:")
        lines += [f"- {e}" for e in push_up]
    if push_down:
        lines.append("Faktor penahan churn:")
        lines += [f"- {e}" for e in push_down]
    return lines


def main() -> None:
    comp = pd.read_csv(BASE / "reports" / "model_comparison.csv").iloc[0]
    model_name = comp["model"]
    print(f"explaining model: {model_name}")
    model = joblib.load(BASE / "models" / f"{model_name}.joblib")

    X_train, X_test, y_train, y_test, feature_names, _ = prepare_data()

    # Raw test rows for readable customer context (same split -> aligned)
    df = clean(load_raw())
    _, X_test_raw = train_test_split(
        df.drop(columns=["target"]), test_size=0.2,
        stratify=df["target"].values, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"shap values shape: {np.shape(shap_values)}")

    # Global importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": [pretty_feature(f) for f in feature_names],
                        "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(BASE / "reports" / "shap_importance.csv", index=False)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2a9d8f")
    ax.set_title("Global feature importance (mean |SHAP|)")
    ax.set_xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=120)
    plt.close()
    print("top-5 global:", ", ".join(imp["feature"].head(5)))

    # Local explanations for two illustrative customers
    proba = model.predict_proba(X_test)[:, 1]
    churners = np.where((y_test == 1) & (proba > 0.7))[0]
    loyal = np.where((y_test == 0) & (proba < 0.2))[0]
    examples = []
    if len(churners):
        examples.append(("high_risk_churner", churners[0]))
    if len(loyal):
        examples.append(("low_risk_customer", loyal[0]))

    md = ["# Contoh Penjelasan SHAP per Pelanggan", ""]
    for label, idx in examples:
        exp = shap.Explanation(values=shap_values[idx],
                               base_values=explainer.expected_value,
                               data=X_test[idx],
                               feature_names=feature_names)
        fig = plt.figure()
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_waterfall_{label}.png", dpi=120,
                    bbox_inches="tight")
        plt.close()

        raw = X_test_raw.iloc[idx]
        md.append(f"## {label} (baris uji #{idx})")
        md.append(f"- Label aktual: {'churn' if y_test[idx] == 1 else 'tidak churn'}")
        md.append(f"- Kontrak: {raw.get('Contract', '-')}, "
                  f"tenure: {raw.get('tenure', '-')}, "
                  f"MonthlyCharges: {raw.get('MonthlyCharges', '-')}")
        md += narrative(pd.Series(X_test[idx], index=feature_names),
                        shap_values[idx], feature_names, proba[idx])
        md.append("")
        print(f"explained {label}: proba={proba[idx]:.1%}")

    (BASE / "reports" / "shap_examples.md").write_text("\n".join(md))
    print("wrote reports/shap_importance.csv, reports/shap_examples.md, figures/shap_*.png")


if __name__ == "__main__":
    main()
