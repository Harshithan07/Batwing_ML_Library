# Getting Started

This guide walks you through a complete workflow with Batwing-ML – starting from a raw CSV and ending with a tuned, evaluated model.

You can copy this into a notebook and run it top to bottom, or adapt each step into your own pipeline.

---

## 1. Installation

First, install Batwing-ML from PyPI.

```bash
pip install batwing-ml
```

Batwing-ML builds on the usual Python data stack. If you’re in a fresh environment, it’s safe to also install:

```bash
pip install pandas numpy scikit-learn optuna
```

* **pandas / numpy** – DataFrame and numerical operations.
* **scikit-learn** – Models, metrics, and utilities.
* **optuna** – Used under the hood for hyperparameter tuning.

---

## 2. Load your data

Start with your dataset as a pandas DataFrame. This can come from CSV, SQL, Parquet, etc.

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

For this walkthrough, we’ll assume your target column is called `label` and all other columns are features.

* You don’t need to clean anything manually yet.
* Missing values, mixed types, and messy columns are expected at this stage.

---

## 3. Validate & clean

Before modeling, it’s important to know **what’s wrong with your data** and to fix simple issues systematically.

```python
from batwing_ml import validate_and_clean_data

df_clean, audit = validate_and_clean_data(df)
```

What this does:

* Standardizes dtypes where possible.
* Flags and optionally drops obviously broken columns (empty, constant, etc.).
* Handles basic range / consistency checks.
* Returns **two things**:

  * `df_clean` – the cleaned DataFrame you will use downstream.
  * `audit` – a structured dict describing what changed.

You can inspect `audit` to see, for example:

* Which columns were dropped or fixed.
* How many rows were affected by cleaning.
* Any schema issues that might need manual review.

This makes your preprocessing **auditable** instead of “we tweaked stuff in a notebook and forgot what happened.”

---

## 4. Quick EDA

Next, get a compact overview of your cleaned data. Batwing-ML includes helpers that produce notebook-friendly summaries.

```python
from batwing_ml import summary_dataframe

summary = summary_dataframe(df_clean)
summary  # renders nicely in notebooks
```

Typical things you’ll see in `summary`:

* Column types (numeric, categorical, datetime, etc.).
* Missing value counts and proportions.
* Basic stats (min, max, mean, std) for numeric columns.
* Cardinality and top values for categoricals.

This gives you a **quick sanity check** before you dive into modeling.

---

## 5. Preprocess and encode

Now we turn the cleaned DataFrame into a model-ready feature matrix: imputed, encoded, and scaled.

```python
from batwing_ml import preprocess_dataframe

X_processed, prep_steps = preprocess_dataframe(
    df_clean.drop(columns=["label"]),
    encode="onehot",
    scale="standard",
    drop_missing_thresh=0.3,
    return_steps=True,
)

y = df_clean["label"]
```

What each argument does:

* `df_clean.drop(columns=["label"])` – use all non-target columns as features.
* `encode="onehot"` – one-hot encode categorical features.
* `scale="standard"` – standardize numeric features (zero mean, unit variance).
* `drop_missing_thresh=0.3` – drop columns with more than 30% missing values.
* `return_steps=True` – return a metadata object describing what preprocessing was applied.

Outputs:

* `X_processed` – a numerical feature matrix ready for modeling.
* `prep_steps` – a record of encoders, scalers, dropped columns, etc. (useful for debugging or exporting your pipeline).

---

## 6. Feature engineering

With a clean feature matrix, you can optionally apply feature engineering and selection.

```python
from batwing_ml import feature_engineering

X_fe, fe_meta = feature_engineering(
    X_processed,
    target=None,
    mode="both",
)
```

Typical things this step can do (depending on configuration):

* Remove low-variance or redundant features.
* Apply supervised selection strategies (e.g., mutual information, model-based importance) when a target is provided.
* Create transformed or composite features.

In this minimal example:

* `mode="both"` tells Batwing-ML to run both selection and transformation logic (where applicable).
* `X_fe` is the final engineered feature matrix.
* `fe_meta` records **what was selected or generated**, so you can inspect or log it later.

If you prefer to keep things simple at first, you can skip this step and use `X_processed` directly.

---

## 7. Train and evaluate a model

Finally, we split the data, tune a model, and evaluate it using Batwing-ML’s tuning and evaluation helpers.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from batwing_ml import (
    hyperparameter_tuning_classification,
    evaluate_classification_model,
)

# 1) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_fe, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 2) Hyperparameter tuning

tuning = hyperparameter_tuning_classification(
    X=X_train,
    y=y_train,
    model_class=RandomForestClassifier,
    param_grid={
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
        "max_depth":    lambda t: t.suggest_int("max_depth", 3, 20),
    },
    scoring="f1_macro",
    fast_mode=True,
)

best_model = tuning["best_model"]

# 3) Final evaluation

metrics = evaluate_classification_model(
    model=best_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    return_dict=True,
)

print(metrics)
```

What’s happening here:

1. **Train/test split** – We keep 20% of the data as a hold-out test set and use stratification to preserve label distribution.
2. **Hyperparameter tuning** –

   * `hyperparameter_tuning_classification` wraps Optuna.
   * `param_grid` defines the search space (here, number of trees and max depth for a random forest).
   * `scoring="f1_macro"` optimizes macro-averaged F1, which is good for imbalanced multiclass problems.
   * `fast_mode=True` can enable smaller subsets or fewer folds for quicker iteration.
3. **Evaluation** – `evaluate_classification_model` computes metrics on both train and test sets and can optionally generate plots (confusion matrix, ROC/PR curves, etc., depending on configuration).

The printed `metrics` dict gives you a compact summary you can log or compare across runs.

---

## 8. Where to go next

From here, you can:

* Swap in different models (e.g., gradient boosting, linear models, etc.).
* Use the **regression** or **multiclass** workflows if your task isn’t simple binary classification.
* Explore the **Guides** section for:

  * A deeper classification workflow.
  * Regression pipelines.
  * Handling large datasets and using fast-mode effectively.
  * Notebook-focused visual previews and reporting.

Batwing-ML is meant to be **composable**: you can keep this exact structure or plug individual steps into your own experiment scripts, pipelines, or MLflow runs.
