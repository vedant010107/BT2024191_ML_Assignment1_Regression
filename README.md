# Polynomial Regression - BT2024191

## Project overview

This project predicts two continuous targets using polynomial regression. Q1 models the Net Power Score of a steam turbine from six operating parameters. Q2 models the Thermal Anomaly Score of a geothermal reservoir from three spatial coordinates. The datasets are independent, and a separate model is fitted for each question using the shared implementation in `main.py`.

Each dataset contains 1,000 labelled training rows and 1,000 test rows. The prediction files contain one `y` column in the original test-row order.

## Preprocessing and model search

Original inputs are standardised, expanded into polynomial features, and standardised again before regression. All monomials up to the selected total degree are included, along with interactions. The intercept is fitted separately. Each scaler is fitted only on the fitting portion of its current fold.

All degrees from 1 to 10 are searched for Q1, and from 1 to 20 for Q2. Each degree has 24 configurations:

| Model | Alpha values | L1 ratio |
| --- | --- | --- |
| Ridge | 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000 | 0 |
| Lasso | 1, 0.1, 0.01, 0.001 | 1 |
| Elastic Net | 1, 0.1, 0.01, 0.001 | 0.2, 0.5, 0.8 |

Selection uses the smallest eligible mean inner-validation MSE. Sparse fits must satisfy a dual-gap convergence check in every relevant fold; failed candidates are recorded and excluded. The search covers a fixed grid and does not assume that error is a convex function of degree.

## Validation procedure

Five grouped outer folds evaluate the model-selection procedure. Three grouped inner folds screen every degree and configuration within each outer training partition. Identical input vectors stay together to avoid duplicate-location leakage. Outer scores are not used to select the final configuration.

For Q1, each outer split has 800 fitting rows and 200 held-out rows. Its inner split has approximately 533 fitting rows and 267 validation rows. Q2 has 996 distinct input locations, resulting in outer held-out sizes of 200, 201, 199, 201 and 199.

Q2 adds repeated validation after screening. The best two distinct degrees from each family form a six-configuration shortlist within the current training partition. Five-fold validation repeated twice ranks that shortlist, with approximately 640 fitting and 160 validation rows in each outer-training partition. Screening uses seed 42, outer evaluation uses seed 43, and repeated validation uses seeds 142 and 143.

Final selection is repeated on all 1,000 labelled rows, after which the selected model is fitted on all labelled rows. The supplied test data are used only for inference.

## Results

| Metric | Q1 | Q2 |
| --- | --- | --- |
| Final model | Lasso | Ridge |
| Polynomial degree | 5 | 10 |
| Alpha | 0.01 | 1 |
| Polynomial terms | 461 | 285 |
| Nonzero coefficients | 120 | 285 |
| Pooled outer-validation MSE | 0.315281 | 0.308099 |
| Pooled outer-validation R-squared | 0.970134 | 0.993064 |
| Full-data inner-selection MSE | 0.336027 | 0.283795 |
| Final training MSE | 0.229104 | 0.180240 |

Pooled outer metrics use the held-out predictions for all 1,000 labelled rows. They estimate the performance of the selection procedure. Final-selection scores are used for choosing the model and are not independent evaluation results. Regularisation and nested validation are used to control overfitting.

The degree plots show full-data three-fold screening scores. For Q2, the selected-degree marker comes from the subsequent repeated-validation stage. Each family curve uses its best eligible configuration at that degree. Convergence exclusions can affect the curves.

## Folder structure

| Path | Contents |
| --- | --- |
| `main.py` | Shared preprocessing, search, validation, fitting, prediction, plotting and numerical checks |
| `requirements.txt` | Python dependency versions |
| `BT2024191/` | Original training and test datasets |
| `ML_Assignment_1.pdf` | Assignment specification |
| `sample_submission.csv` | Prediction-file format reference |
| `Q1/`, `Q2/` | Models, predictions and supporting results for each question |
| `Report/` | Five-page PDF report |
| `terminal_output.txt` | Detailed output from both completed runs |

Each question folder contains:

- `model.joblib`: fitted scalers, polynomial transformation and regression coefficients.
- `BT2024191_pred_var1.csv` or `BT2024191_pred_var2.csv`: final test predictions.
- `final_model.json`: selected configuration, training MSE and selection MSE.
- `nested_cv.json` and `validation_predictions.csv`: outer-fold metrics and held-out predictions.
- `degree_mse.csv` and `images/`: degree-wise results and graphs.
- `search/`: candidate scores, convergence records and inner-fold assignments.
- `outer_splits.json`: outer fitting and evaluation row indices.
- `manifest.json` and `logging_compatibility.json`: data, settings, version and implementation checks for checkpoint reuse.
- `verification.json` and `RESULTS.md`: validation checks and question-specific results.

## Execution behaviour

A normal run processes both questions. It prints degree-wise validation MSE, invalid-candidate counts, outer-fold MSE and R-squared, Q2 shortlist scores, and a final summary. Compatible completed results are labelled `[cached]` and verified without repeating training. Interrupted runs resume from checkpoints. Incompatible code, data, settings or package versions cause checkpoint reuse to be rejected.

A fresh run without existing question-output folders performs the complete search. Saved models support inference without repeating selection. The numerical check compares regression solutions against independent reference solvers and checks preprocessing and duplicate separation.

## Commands

All commands below run from the Project directory with the project environment active.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run both questions:

```powershell
python main.py
```

Run one question:

```powershell
python main.py --question 1
python main.py --question 2
```

Generate predictions from the saved models:

```powershell
python main.py --action predict
```

Regenerate graphs:

```powershell
python main.py --action plot
```

Run numerical checks:

```powershell
python main.py --action check
```

Set the number of parallel degree-search jobs:

```powershell
python main.py --jobs 2
```

