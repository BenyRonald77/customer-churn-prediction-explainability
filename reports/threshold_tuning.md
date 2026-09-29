# Threshold Tuning

Model: `xgb_balanced` (imbalance: class_weight)

## Metrics per threshold

|   threshold |   precision |   recall |     f1 |
|------------:|------------:|---------:|-------:|
|        0.1  |      0.3751 |   0.9599 | 0.5394 |
|        0.15 |      0.4007 |   0.9332 | 0.5606 |
|        0.2  |      0.4236 |   0.9118 | 0.5785 |
|        0.25 |      0.4436 |   0.893  | 0.5927 |
|        0.3  |      0.4629 |   0.8663 | 0.6034 |
|        0.35 |      0.4785 |   0.8342 | 0.6082 |
|        0.4  |      0.5    |   0.8182 | 0.6207 |
|        0.45 |      0.515  |   0.7807 | 0.6206 |
|        0.5  |      0.5324 |   0.746  | 0.6214 |
|        0.55 |      0.5412 |   0.7032 | 0.6116 |
|        0.6  |      0.5605 |   0.6684 | 0.6098 |
|        0.65 |      0.5887 |   0.6123 | 0.6003 |
|        0.7  |      0.6341 |   0.5561 | 0.5926 |
|        0.75 |      0.682  |   0.4759 | 0.5606 |
|        0.8  |      0.7206 |   0.393  | 0.5087 |
|        0.85 |      0.7704 |   0.2781 | 0.4086 |
|        0.9  |      0.8101 |   0.1711 | 0.2826 |

## Recommended thresholds
- Best F1: threshold=0.50 (P=0.5324, R=0.746, F1=0.6214)
- Best F1 with recall>=0.75: threshold=0.40 (P=0.5, R=0.8182, F1=0.6207)

Note: default 0.5 is rarely optimal on imbalanced data. Pick the threshold that matches the business cost of missing a churner.