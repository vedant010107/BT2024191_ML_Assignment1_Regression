# Q1 results

Status: complete.

| Item | Result |
|---|---:|
| Selected family | Lasso |
| Degree | 5 |
| Alpha | 0.01 |
| Outer pooled MSE | 0.315281 |
| Outer pooled R-squared | 0.970134 |
| Full-data inner selection MSE | 0.336027 |
| Full-data training MSE | 0.229104 |
| Nonzero coefficients / polynomial terms | 120 / 461 |

All five outer folds selected degree-5 Lasso with alpha 0.01. Their MSEs were:

- Fold 1: 0.321445
- Fold 2: 0.266525
- Fold 3: 0.320801
- Fold 4: 0.395805
- Fold 5: 0.271830

## Search and checks

Every degree from 1 to 10 was tested with 24 Ridge/Lasso/Elastic Net configurations. Five outer-fold searches and one full-data search produced 4,320 inner candidate fits. 329 inner fits failed the numerical convergence check; any candidate failing an inner fold was excluded for that search. Detailed candidate scores and failures remain in search/.

The outer split is 800 fitting / 200 held-out rows. Within each outer training partition, three-fold selection uses approximately 533 fitting / 267 validation rows. Preprocessing is fitted within each inner training fold. Final fitting uses all 1,000 labelled rows after selection.

Training MSE is lower than outer MSE. Validation error is lowest at degree 5 in the full-data screen and rises at larger degrees. The stable fold selections are reassuring but do not prove absence of overfitting. Different candidate convergence outcomes can also affect the curves. These are internal cross-validation estimates.

The prediction CSV contains 1,000 finite values in the original test order. Reloaded model predictions match it. Code/data hashes, full degree coverage, saved outer MSE and nested fold separation were verified.

## Plots

- images/degree_mse.png: best eligible validation MSE per degree/family; training and validation errors for the best eligible configuration at each degree. The y-axis is logarithmic and the vertical line marks degree 5.
- images/validation_predictions.png: outer-fold predictions versus actual targets.
- degree_mse.csv: underlying degree/family table.

Q2 and the final combined report are also complete.
