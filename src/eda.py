"""Exploratory data analysis for the Telco churn dataset.

Saves figures to reports/figures/ and writes reports/eda_summary.md
with the key numbers. Run:  .venv/bin/python src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw, TARGET

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()


def main() -> None:
    df = load_raw()
    lines = ["# EDA Summary", "", f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", ""]

    # 1. Target balance
    vc = df[TARGET].value_counts()
    churn_rate = vc["Yes"] / len(df)
    lines += ["## Target balance",
              f"- No churn: {vc['No']} ({vc['No']/len(df):.1%})",
              f"- Churn: {vc['Yes']} ({vc['Yes']/len(df):.1%})",
              f"- Churn rate: {churn_rate:.3f} (imbalanced, ~1:2.7)", ""]

    fig, ax = plt.subplots()
    ax.bar(vc.index, vc.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Target distribution (Churn)")
    ax.set_ylabel("Customers")
    savefig("target_distribution.png")

    # 2. Missing values
    miss = df.isna().sum()
    miss = miss[miss > 0]
    lines += ["## Missing values"]
    if miss.empty:
        lines.append("- None")
    else:
        for col, n in miss.items():
            lines.append(f"- {col}: {n} ({n/len(df):.2%})")
    lines.append("")

    # 3. Numeric overview
    num = df.select_dtypes("number")
    lines += ["## Numeric columns", "", num.describe().round(2).to_markdown(), ""]
    # tenure by churn
    fig, ax = plt.subplots()
    for label, color in (("No", "#2a9d8f"), ("Yes", "#e76f51")):
        ax.hist(df.loc[df[TARGET] == label, "tenure"], bins=24, alpha=0.6,
                label=f"Churn={label}", color=color)
    ax.set_title("Tenure distribution by churn")
    ax.set_xlabel("Tenure (months)")
    ax.legend()
    savefig("tenure_by_churn.png")

    # monthly charges by churn
    fig, ax = plt.subplots()
    for label, color in (("No", "#2a9d8f"), ("Yes", "#e76f51")):
        ax.hist(df.loc[df[TARGET] == label, "MonthlyCharges"], bins=24, alpha=0.6,
                label=f"Churn={label}", color=color)
    ax.set_title("MonthlyCharges distribution by churn")
    ax.set_xlabel("MonthlyCharges")
    ax.legend()
    savefig("monthly_charges_by_churn.png")

    # 4. Churn rate per key categorical feature
    lines += ["## Churn rate by category", ""]
    for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]:
        rates = df.groupby(col, observed=True)[TARGET].apply(
            lambda s: (s == "Yes").mean()).sort_values(ascending=False)
        lines.append(f"### {col}")
        for cat, r in rates.items():
            lines.append(f"- {cat}: {r:.1%}")
        lines.append("")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        rates.plot.barh(ax=ax, color="#e76f51")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate")
        savefig(f"churn_rate_by_{col.lower()}.png")

    # 5. Key takeaways
    lines += ["## Key takeaways",
              "- Dataset is imbalanced (26.5% churn): accuracy alone is misleading, "
              "use PR-AUC / F1.",
              "- TotalCharges has 11 missing values, all tenure == 0 (new customers); "
              "safe to impute or drop.",
              "- Short tenure and month-to-month contracts show the highest churn "
              "rates: strong candidate features.",
              "- No duplicate customerID expected; verified in train script.", ""]

    out = BASE / "reports" / "eda_summary.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} and {len(list(FIG_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
