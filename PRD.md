# PRD: Customer Churn Prediction + Explainability

**Proyek:** customer-churn-prediction-explainability
**Pemilik:** BenyRonald77
**Status:** Draft awal
**Tanggal:** 2026-09-29

## 1. Latar Belakang

Churn pelanggan adalah masalah klasik dengan dampak bisnis langsung: mempertahankan
pelanggan lama jauh lebih murah daripada mengakuisisi yang baru. Proyek ini membangun
pipeline machine learning end-to-end untuk memprediksi pelanggan mana yang berisiko
churn, dengan penekanan pada dua hal yang sering diabaikan tutorial: penanganan data
tidak seimbang dan kemampuan menjelaskan setiap prediksi ke level individu pelanggan.

## 2. Tujuan

1. Membangun model klasifikasi churn dengan performa terukur dan dapat direproduksi.
2. Menangani ketidakseimbangan kelas secara eksplisit (SMOTE dan class weights),
   bukan sekadar mengandalkan akurasi.
3. Membandingkan beberapa algoritma secara adil dengan protokol evaluasi yang sama.
4. Menjelaskan prediksi per pelanggan menggunakan SHAP, sehingga hasilnya bisa
   ditindaklanjuti tim bisnis (alasan churn, bukan sekadar skor).
5. Menyediakan artefak siap pakai: model tersimpan, skrip inferensi, dan dashboard
   sederhana untuk eksplorasi prediksi.

## 3. Ruang Lingkup

### Masuk lingkup

- Dataset: Telco Customer Churn (IBM). Sumber utama: mirror CSV publik IBM di GitHub.
  Fallback bila tidak dapat diunduh: generator data sintetis terdokumentasi dengan
  skema fitur yang sama.
- Eksplorasi data (EDA): distribusi kelas, missing values, korelasi fitur.
- Preprocessing: encoding kategorikal, scaling numerik, train/test split stratified.
- Penanganan imbalance: SMOTE pada data latih dan class weights pada model,
  dibandingkan secara eksplisit.
- Model yang dibandingkan: Logistic Regression (baseline), Random Forest,
  XGBoost. Minimal tiga model.
- Evaluasi: precision, recall, F1, ROC-AUC, dan kurva precision-recall (metrik utama
  untuk data imbalanced, bukan akurasi). Threshold tuning berdasarkan F1 dan
  kebutuhan recall.
- Explainability: SHAP global (feature importance) dan SHAP lokal per pelanggan
  (waterfall/force plot) beserta narasi alasan churn dalam bahasa manusia.
- Artefak: model pipeline tersimpan (joblib), skrip `predict.py` untuk inferensi
  satu pelanggan dari CLI.
- Dashboard Streamlit: input data pelanggan, tampilkan probabilitas churn dan
  penjelasan SHAP per pelanggan.

### Di luar lingkup

- Data real-time atau streaming; proyek ini batch/offline.
- Deployment produksi (API server, CI/CD, monitoring drift). Artefak disiapkan
  agar mudah dibungkus API bila dibutuhkan nanti.
- Hyperparameter search ekstensif (grid search penuh). Tuning dibatasi pada
  parameter kunci tiap model agar waktu latih wajar.

## 4. Fungsionalitas dan Tahapan Build

Setiap tahap di bawah ini dikerjakan berurutan dan di-commit + push sebagai satu
commit tersendiri.

| Tahap | Fungsionalitas | Output |
|-------|----------------|--------|
| F0 | Fondasi proyek: struktur direktori, README, .gitignore, requirements | Commit fondasi |
| F1 | Data loading + EDA (script/notebook, ringkasan temuan) | `src/data_loader.py`, `notebooks/01_eda.ipynb`, laporan EDA |
| F2 | Preprocessing + penanganan imbalance (SMOTE, class weights) | `src/preprocess.py`, perbandingan strategi sampling |
| F3 | Training + perbandingan model (LogReg, RF, XGBoost) | `src/train.py`, tabel perbandingan, model tersimpan |
| F4 | Evaluasi + threshold tuning (precision-recall) | `src/evaluate.py`, kurva PR/ROC, threshold terpilih |
| F5 | Explainability SHAP global dan per pelanggan | `src/explain.py`, plot SHAP, contoh narasi per pelanggan |
| F6 | Skrip inferensi CLI + artefak final | `predict.py`, `models/` final, dokumentasi pemakaian |
| F7 | Dashboard Streamlit untuk eksplorasi prediksi | `app.py`, berjalan lokal via `streamlit run app.py` |

## 5. Metrik Keberhasilan

- ROC-AUC model terbaik >= 0.80 pada test set (target realistis untuk dataset ini).
- PR-AUC dilaporkan sebagai metrik utama; akurasi tidak dipakai sebagai acuan.
- Setiap strategi imbalance (tanpa penanganan, SMOTE, class weights) punya angka
  pembanding pada model yang sama.
- Minimal satu contoh penjelasan SHAP per pelanggan yang bisa dibaca non-teknis.
- Seluruh pipeline bisa dijalankan ulang dari nol dengan satu perintah per tahap
  dan menghasilkan artefak yang sama (seed tetap).

## 6. Tech Stack

- Python 3.12
- pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib,
  streamlit, joblib
- Semua dependency dipin di `requirements.txt`

## 7. Struktur Direktori (rencana)

```
.
├── PRD.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # dataset asli (tidak di-commit bila besar)
│   └── processed/      # hasil preprocessing
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── models/             # pipeline model tersimpan (.joblib)
├── reports/
│   └── figures/        # plot evaluasi dan SHAP
├── predict.py
└── app.py              # dashboard Streamlit
```

## 8. Standar Kualitas

- Setiap skrip bisa dijalankan mandiri dan punya `--help` atau docstring yang jelas.
- Seed acak ditetapkan di semua tahap (reproduksibilitas).
- Tidak ada angka/metrik yang diklaim tanpa output evaluasi yang tersimpan.
- Mode antislop: DURING. Aturan antislop + antislop-ui diterapkan selama
  pengerjaan, terutama pada dashboard (F7). Sebelum membangun UI pada F7,
  direction desain disepakati dulu (R-37): bila belum ada DESIGN.md, opsi
  diminta ke pemilik sebelum mulai, dan Delivery Gate dijalankan sebelum
  dashboard dinyatakan selesai.

## 9. Risiko dan Mitigasi

- Dataset publik tidak bisa diunduh: gunakan generator sintetis terdokumentasi.
- SHAP lambat pada model tree besar: batasi background sample dan gunakan
  TreeExplainer.
- Waktu latih XGBoost lama di VM kecil: batasi jumlah estimator dan pakai
  early stopping.

## 10. Definition of Done

Proyek dinyatakan selesai bila: seluruh tahap F0-F7 ter-commit dan ter-push,
README menjelaskan cara menjalankan tiap tahap, model terbaik terdokumentasi
beserta metriknya, minimal satu penjelasan SHAP per pelanggan tersedia di
dashboard, dan tidak ada langkah manual yang tidak tertulis.
