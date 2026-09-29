# Contoh Penjelasan SHAP per Pelanggan

## high_risk_churner (baris uji #9)
- Label aktual: churn
- Kontrak: Month-to-month, tenure: 17, MonthlyCharges: 45.05
Probabilitas churn model: 73.4%

Faktor pendorong churn:
- Contract Month-to-month = 1.00 (+0.493)
- PaymentMethod Electronic check = 1.00 (+0.333)
- MonthlyCharges = -0.66 (+0.225)
- SeniorCitizen = 2.26 (+0.120)
Faktor penahan churn:
- InternetService Fiber optic = 0.00 (-0.318)

## low_risk_customer (baris uji #0)
- Label aktual: tidak churn
- Kontrak: Two year, tenure: 72, MonthlyCharges: 114.05
Probabilitas churn model: 0.6%

Faktor penahan churn:
- tenure = 1.61 (-1.390)
- Contract Month-to-month = 0.00 (-1.083)
- Contract Two year = 1.00 (-0.828)
- Contract One year = 0.00 (-0.370)
- TotalCharges = 2.71 (-0.286)
