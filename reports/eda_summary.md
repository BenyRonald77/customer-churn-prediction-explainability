# EDA Summary

Rows: 7043, Columns: 21

## Target balance
- No churn: 5174 (73.5%)
- Churn: 1869 (26.5%)
- Churn rate: 0.265 (imbalanced, ~1:2.7)

## Missing values
- TotalCharges: 11 (0.16%)

## Numeric columns

|       |   SeniorCitizen |   tenure |   MonthlyCharges |   TotalCharges |
|:------|----------------:|---------:|-----------------:|---------------:|
| count |         7043    |  7043    |          7043    |        7032    |
| mean  |            0.16 |    32.37 |            64.76 |        2283.3  |
| std   |            0.37 |    24.56 |            30.09 |        2266.77 |
| min   |            0    |     0    |            18.25 |          18.8  |
| 25%   |            0    |     9    |            35.5  |         401.45 |
| 50%   |            0    |    29    |            70.35 |        1397.48 |
| 75%   |            0    |    55    |            89.85 |        3794.74 |
| max   |            1    |    72    |           118.75 |        8684.8  |

## Churn rate by category

### Contract
- Month-to-month: 42.7%
- One year: 11.3%
- Two year: 2.8%

### InternetService
- Fiber optic: 41.9%
- DSL: 19.0%
- No: 7.4%

### PaymentMethod
- Electronic check: 45.3%
- Mailed check: 19.1%
- Bank transfer (automatic): 16.7%
- Credit card (automatic): 15.2%

### PaperlessBilling
- Yes: 33.6%
- No: 16.3%

## Key takeaways
- Dataset is imbalanced (26.5% churn): accuracy alone is misleading, use PR-AUC / F1.
- TotalCharges has 11 missing values, all tenure == 0 (new customers); safe to impute or drop.
- Short tenure and month-to-month contracts show the highest churn rates: strong candidate features.
- No duplicate customerID expected; verified in train script.
