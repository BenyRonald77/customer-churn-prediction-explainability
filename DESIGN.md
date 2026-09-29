# DESIGN.md — Dashboard "Model Lab" Churn

Keputusan desain tertulis (anti-slop). Dashboard ini adalah **model lab**:
pekerjaannya adalah menjawab "model mana yang menang, threshold berapa yang dipakai,
dan apa pendorong churn" — bukan memonitor operasional.

## Struktur mengikuti narasi, bukan template dashboard generik

1. **Verdict** — satu baris kesimpulan (model terbaik + PR-AUC) di atas.
2. **Perbandingan model** — tabel nyata dari `reports/model_comparison.csv` + bar chart PR-AUC.
   Pertanyaan: "model mana yang menang dan selisihnya berapa?"
3. **Threshold tuning** — slider interaktif pada model terbaik, metrik P/R/F1 live.
   Pertanyaan: "threshold berapa yang kita ship?"
4. **Kurva PR & ROC** — figure top-3 dari `reports/figures/curves_top3.png`.
   Pertanyaan: "seberapa bagus ranking model di seluruh threshold?"
5. **SHAP global** — `reports/figures/shap_summary.png` + tabel top-15.
   Pertanyaan: "fitur apa yang mendorong churn?"

Tidak ada: sidebar navigasi multi-halaman (satu layar cukup), stat cards deretan
angka (hanya satu verdict), activity feed, tabel generik Name/Status/Date.

## Palet (satu aksen, tema terang)

- Base: putih hangat `#fafaf9`, teks charcoal `#1c1917`, garis `#e7e5e4`.
- Aksen tunggal: teal tua `#0f766e` — dipakai hanya untuk: bar chart PR-AUC,
  penanda model terbaik, dan highlight threshold. Alasan: teal = "analitis/tenang",
  kontras cukup terhadap teks gelap, dan konsisten dengan figure SHAP yang sudah
  dibuat (`#2a9d8f`).
- Tidak ada gradient, glow, glassmorphism, atau dark mode (produk content-first,
  tidak ada alasan brand untuk dark).

## Tipografi & komponen

- Font default Streamlit (alasan: dashboard internal, bukan marketing page;
  memilih font custom tanpa identitas brand = dekorasi).
- Radius kecil dan konsisten; shadow hanya untuk kartu verdict (satu elevasi).
- Tanpa emoji di teks UI; tanpa badge pil "AI Powered".

## Data: semua angka nyata

- Tabel dan chart membaca langsung dari `reports/*.csv` dan `reports/figures/*.png`
  yang dihasilkan pipeline. Tidak ada angka placeholder.
- Threshold slider menghitung ulang P/R/F1 dari probabilitas test set yang nyata.
- State kosong: jika artefak belum ada (mis. `train.py` belum dijalankan),
  tampilkan pesan penyebab + perintah yang harus dijalankan (bukan "No data").

## Dials

- RHYTHM: 2 — seksi bergantian antara tabel/chart dan teks (tidak monoton).
- MOTION: 1 — hanya hover states bawaan Streamlit, tanpa animasi loop.
