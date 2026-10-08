# Q2 results

Status: complete.

| Item | Result |
|---|---:|
| Selected family | Ridge |
| Degree | 10 |
| Alpha | 1 |
| Outer pooled MSE | 0.308099 |
| Outer pooled R-squared | 0.993064 |
| Mean outer-fold MSE | 0.307842 |
| Outer-fold MSE standard deviation | 0.081575 |
| Full-data repeated inner-selection MSE | 0.283795 |
| Full-data training MSE | 0.180240 |
| Polynomial terms | 285 |

## Outer evaluation

| Fold | Selected family | Degree | Alpha | L1 ratio | Training MSE | Held-out MSE |
|---|---|---:|---:|---:|---:|---:|
| 1 | ridge | 10 | 1.0 | 0.0 | 0.166280 | 0.306633 |
| 2 | lasso | 9 | 0.001 | 1.0 | 0.190858 | 0.438271 |
| 3 | ridge | 10 | 1.0 | 0.0 | 0.185867 | 0.217187 |
| 4 | elastic_net | 10 | 0.001 | 0.5 | 0.175701 | 0.306490 |
| 5 | ridge | 11 | 1.0 | 0.0 | 0.165571 | 0.270628 |

Outer held-out sizes were 200, 201, 199, 201 and 199. Pooled MSE weights every observation equally; the unweighted average of fold MSEs therefore differs slightly. Fold standard deviation is descriptive, not a formal confidence interval.

## Search and validation

Every degree from 1 to 20 was screened using 24 Ridge/Lasso/Elastic Net configurations and three grouped inner folds. For each partition, the best two distinct degrees per family formed a fixed six-candidate shortlist. That shortlist was reranked with five grouped inner folds repeated twice. It was formed using only that partition's training rows. Scaling, polynomial fitting and regression stayed inside each fitting fold.

The five outer evaluations and final full-data selection completed 8,640 screening fits and 360 repeated-inner fits. There were 933 nonconvergent screening fits and 11 nonconvergent repeated-inner fits. A candidate was eligible only if it converged in every relevant inner fold. All candidate scores, failed fits and exact row assignments are preserved in search/.

The data contain 1,000 rows at 996 unique input locations. Identical locations stayed together in all splits. Each outer fit uses approximately 800 rows and evaluates on approximately 200. Repeated inner fitting/validation uses approximately 640/160 of those outer-training rows. Final fitting uses all 1,000 labelled rows.

## Interpretation and limitations

The final all-data selection chose degree-10 Ridge with alpha 1. Outer choices varied: three Ridge models, one Lasso and one Elastic Net. Training errors are lower than held-out errors, and fold errors vary. Regularisation and nested validation are used to control overfitting.

The selected Ridge alpha is inside its searched range. Some shortlisted sparse-model alphas are at the lower grid boundary. This is a fixed finite search, not proof of a globally optimal degree or penalty. Convergence exclusions also affect the high-degree curves.

## Graphs and outputs

- images/degree_mse.png shows the three-fold full-data SCREENING scores, not repeated-inner scores. Each family curve uses its best eligible alpha per degree. The right panel uses the best eligible screening configuration at each degree. The vertical line marks the final degree chosen after repeated validation. Axes use logarithmic MSE.
- images/validation_predictions.png shows pooled outer-fold predictions versus actual targets.
- degree_mse.csv contains the degree/family screening table.
- search/full/repeated_selection.json contains the final six candidates' repeated-validation scores.
- model.joblib contains the final fitted model and preprocessing.
- BT2024191_pred_var2.csv contains 1,000 verified finite test predictions, in the original test order.

Code/data hashes, reloaded-model inference, all 65 outer-partition inner splits, duplicate separation and the pooled MSE were verified. The final combined report is available in ../Report/BT2024191_Report.pdf.
