# Employee Attrition: Data Checks and Baseline Models

Predicting which employees are likely to leave, using the public **IBM HR Analytics Employee Attrition** dataset.
The focus is on doing the basics carefully: checking the data before modeling, choosing metrics that fit an imbalanced problem, measuring uncertainty, and being clear about limitations.

## Data

- **Source:** [IBM HR Analytics Employee Attrition & Performance (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). This is a fictional dataset created by IBM data scientists.
- **Size:** 1,470 employees × 35 columns, with no missing values.
- **Target:** `Attrition` (Yes/No). Only **16.1%** of employees left, so the classes are imbalanced.
- Download the CSV and place it at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

## What's in this repo

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Checks data quality and structure: size, target balance, constant and ID columns, missing values, attrition rate by group (with group size), and skew. |
| `02_baseline.ipynb` | Builds two leak-free scikit-learn pipelines (Logistic Regression and LightGBM), evaluates them on a stratified hold-out set and with 5-fold cross-validation, and reviews what drives the predictions. |

## Approach

1. **Data checks:** I dropped 3 constant columns (`EmployeeCount`, `Over18`, `StandardHours`) and the ID column (`EmployeeNumber`). I also confirmed there were no missing values.
2. **Stratified 80/20 split:** this keeps the ~16% attrition rate the same in train and test.
3. **Pipelines:** preprocessing lives inside a scikit-learn `Pipeline`, so it is fit on training data only, with no leakage into test or CV folds.
   - *Logistic Regression:* `log1p` on skewed income/tenure columns, scaling, one-hot encoding, `class_weight="balanced"`.
   - *LightGBM:* one-hot encoding only (trees don't need scaling), `class_weight="balanced"`.
4. **Metrics:** ROC-AUC, PR-AUC, recall and precision instead of accuracy, because a model that predicts "stays" for everyone would already be 84% accurate.
5. **Uncertainty:** the test set has only 47 leavers, so I also report the 5-fold CV mean ± standard deviation.

## Results

| Model | Test ROC-AUC | Test PR-AUC | Recall | Precision | 5-fold CV AUC |
|---|---|---|---|---|---|
| Logistic Regression | **0.802** | **0.561** | **0.660** | 0.392 | **0.833 ± 0.029** |
| LightGBM | 0.780 | 0.510 | 0.447 | **0.538** | 0.810 ± 0.030 |

*PR-AUC for random guessing is about 0.16.*

**Key findings**
- The simpler, explainable **Logistic Regression** model performs best. It catches two-thirds of leavers.
- **Overtime** is the clearest signal: overtime workers leave at about 3× the rate of others (30.5% vs 10.4%).
- Frequent business travel, roles such as Laboratory Technician and Sales Representative, and having worked at many companies push toward leaving. The Research Director role, no business travel, higher income and more working years push toward staying.
- One surprise: `JobLevel` gets a *positive* coefficient (toward leaving), even though senior roles leave less overall. This is likely because job level overlaps heavily with income and working years. It's a good reminder not to read single coefficients on their own.

## Limitations

- **Associations, not causes.** Coefficients show patterns in this data, not why people leave. Some features overlap (job level, income, working years), so individual coefficients should be read with care.
- **Small minority class.** Results from a single split can move a lot, which is why the CV range is reported.
- **Fairness.** Gender and marital status are model inputs. Before any real use, error rates should be compared across groups.
- **Threshold.** The 0.5 cut-off was not tuned. In practice, it should reflect the cost of a missed leaver vs. a false alarm.
- **Fictional data.** The results show the method, not real-world attrition rates.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Run `01_eda.ipynb`, then `02_baseline.ipynb`.

## Author

Fatma Noor · [GitHub](https://github.com/Fatma-hi)
