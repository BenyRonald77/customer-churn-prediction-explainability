# Customer Churn Prediction + Explainability

Pipeline machine learning end-to-end untuk memprediksi churn pelanggan telekomunikasi,
dengan penanganan data tidak seimbang (SMOTE, class weights), perbandingan beberapa
model, dan penjelasan prediksi per pelanggan menggunakan SHAP.

Dataset: Telco Customer Churn (7043 pelanggan, churn rate 26.5%) — tersedia di `data/raw/`.

## Hasil Utama

| Model | Imbalance | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost | class_weight | 0.53 | 0.75 | 0.62 | 0.84 | **0.65** |
| XGBoost | none | 0.65 | 0.51 | 0.57 | 0.84 | 0.64 |
| LogReg | none | 0.66 | 0.56 | 0.60 | 0.84 | 0.64 |

Model terbaik: **XGBoost + class_weight** (threshold 0.40 → recall ≥ 0.75).
Lihat `reports/model_comparison.csv`, `reports/threshold_tuning.md`, dan `reports/shap_examples.md`.

Faktor churn teratas (SHAP): kontrak month-to-month, tenure rendah, MonthlyCharges tinggi,
Fiber optic, TotalCharges.

## Cara Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. EDA
python src/eda.py
# 2. Preprocessing
python src/preprocess.py
# 3. Training + perbandingan model
python src/train.py
# 4. Evaluasi + threshold tuning
python src/evaluate.py
# 5. Explainability SHAP
python src/explain.py
# 6. Prediksi satu pelanggan
python predict.py --json '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":90.0,"TotalCharges":90.0}'
# 7. Dashboard
streamlit run app.py
```

## Struktur

```
├── PRD.md                  # product requirements
├── requirements.txt
├── predict.py              # CLI inferensi satu pelanggan (+ faktor SHAP)
├── app.py                  # dashboard Streamlit
├── src/
│   ├── data_loader.py      # F1: load dataset
│   ├── eda.py              # F1: EDA -> reports/eda_summary.md
│   ├── preprocess.py       # F2: cleaning, encoding, split, SMOTE
│   ├── train.py            # F3: 3 model x 3 strategi imbalance
│   ├── evaluate.py         # F4: kurva PR/ROC, threshold tuning
│   └── explain.py          # F5: SHAP global + per pelanggan
├── notebooks/01_eda.ipynb
├── data/raw/               # dataset (di-commit, <1MB)
├── data/processed/         # parquet hasil preprocessing (dibangun ulang)
├── models/                 # artefak final: preprocessor + xgb_balanced
└── reports/                # ringkasan, tabel, dan figure tiap tahap
```

## Catatan Desain

- Akurasi tidak dipakai untuk seleksi model (data imbalanced); metrik utama PR-AUC/F1.
- Threshold default 0.40 dipilih dari tuning (F1 terbaik dengan recall ≥ 0.75).
- "No phone/internet service" digabung menjadi "No" sebelum one-hot encoding.
