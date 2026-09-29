"""Model Lab dashboard: compare models, tune the threshold, inspect SHAP.

Run:  streamlit run app.py
Requires the pipeline artifacts (run src/train.py, src/evaluate.py, src/explain.py first).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import prepare_data

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
ACCENT = "#0f766e"


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    p = REPORTS / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_threshold_table() -> pd.DataFrame | None:
    # parsed from the markdown report written by src/evaluate.py
    p = REPORTS / "threshold_tuning.md"
    if not p.exists():
        return None
    lines = [l for l in p.read_text().splitlines() if l.startswith("|")]
    if len(lines) < 3:
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    df = pd.DataFrame(rows, columns=header)
    return df.astype(float)


@st.cache_data
def load_shap_importance() -> pd.DataFrame | None:
    p = REPORTS / "shap_importance.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_best_model(name: str):
    return joblib.load(BASE / "models" / f"{name}.joblib")


@st.cache_data
def best_model_probas(model_name: str):
    _, X_test, _, y_test, _, _ = prepare_data()
    proba = load_best_model(model_name).predict_proba(X_test)[:, 1]
    return proba, y_test


def missing_artifacts() -> list[str]:
    needed = {"model_comparison.csv": "src/train.py",
              "threshold_tuning.md": "src/evaluate.py",
              "shap_importance.csv": "src/explain.py"}
    return [f"{f} (jalankan `{cmd}`)" for f, cmd in needed.items()
            if not (REPORTS / f).exists()]


st.set_page_config(page_title="Churn Model Lab", layout="wide")
st.title("Churn Model Lab")
st.caption("Perbandingan model, threshold tuning, dan penjelasan SHAP "
           "untuk prediksi churn pelanggan Telco.")

missing = missing_artifacts()
if missing:
    st.warning("Artefak berikut belum ada:\n\n- " + "\n- ".join(missing))
    st.stop()

comp = load_comparison()
best = comp.iloc[0]

# 1. Verdict
st.markdown(
    f"<div style='border:1px solid #e7e5e4;border-radius:8px;padding:16px;"
    f"box-shadow:0 1px 3px rgba(0,0,0,0.08);'>"
    f"<b>Model terbaik: {best['model']}</b> "
    f"(imbalance: {best['imbalance']}) &nbsp;·&nbsp; "
    f"PR-AUC <b>{best['pr_auc']:.3f}</b> &nbsp;·&nbsp; "
    f"F1 <b>{best['f1']:.3f}</b> @threshold 0.5 &nbsp;·&nbsp; "
    f"ROC-AUC <b>{best['roc_auc']:.3f}</b></div>",
    unsafe_allow_html=True,
)

# 2. Model comparison
st.header("Perbandingan model")
st.dataframe(comp.style.format("{:.4f}", subset=["precision", "recall", "f1",
                                                 "roc_auc", "pr_auc"]),
             use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 3.5))
order = comp.sort_values("pr_auc")
colors = [ACCENT if m == best["model"] else "#d6d3d1" for m in order["model"]]
ax.barh(order["model"], order["pr_auc"], color=colors)
ax.set_xlabel("PR-AUC (test set)")
ax.set_title("PR-AUC per model — metrik utama untuk data imbalanced")
for i, v in enumerate(order["pr_auc"]):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
st.pyplot(fig)

# 3. Threshold tuning (interactive, on the best model)
st.header("Threshold tuning")
proba, y_test = best_model_probas(best["model"])
thr = st.slider("Decision threshold", 0.05, 0.95, 0.40, 0.05)
pred = (proba >= thr).astype(int)
tp = int(((pred == 1) & (y_test == 1)).sum())
fp = int(((pred == 1) & (y_test == 0)).sum())
fn = int(((pred == 0) & (y_test == 1)).sum())
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
c1, c2, c3 = st.columns(3)
c1.metric("Precision", f"{prec:.3f}")
c2.metric("Recall", f"{rec:.3f}")
c3.metric("F1", f"{f1:.3f}")
st.caption(f"Pada threshold {thr:.2f}: {tp} churn tertangkap, {fp} false alarm, "
           f"{fn} churn terlewat (dari {(y_test == 1).sum()} churn aktual).")

ttable = load_threshold_table()
if ttable is not None:
    st.subheader("Grid threshold dari evaluasi offline")
    st.dataframe(ttable.style.format("{:.2f}", subset=["threshold"])
                 .format("{:.4f}", subset=["precision", "recall", "f1"]),
                 use_container_width=True)

# 4. Curves
st.header("Kurva PR & ROC (3 model teratas)")
curves_png = REPORTS / "figures" / "curves_top3.png"
if curves_png.exists():
    st.image(str(curves_png))
st.caption("PR-AUC lebih informatif daripada ROC-AUC saat kelas positif minoritas.")

# 5. SHAP global
st.header("Apa pendorong churn? (SHAP global)")
imp = load_shap_importance()
shap_png = REPORTS / "figures" / "shap_summary.png"
if shap_png.exists():
    st.image(str(shap_png))
st.subheader("Top-15 fitur")
st.dataframe(imp.head(15).style.format("{:.4f}", subset=["mean_abs_shap"]),
             use_container_width=True)
st.caption("Nilai = rata-rata |SHAP| pada test set. "
           "Lihat reports/shap_examples.md untuk penjelasan per pelanggan.")
