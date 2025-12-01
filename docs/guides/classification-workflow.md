# Tabular Classification Workflow

This guide shows the recommended end-to-end flow for classification problems using Batwing-ML.

It assumes you already understand the core pieces from the **Home** and **Getting Started** pages and want a slightly more opinionated recipe for day‑to‑day work.

---

## 1. Load and validate your data

Start from a raw CSV or table and immediately bring it into a clean, auditable state.

```python
import pandas as pd
from batwing_ml import validate_and_clean_data

df = pd.read_csv("data.csv")

# Validate schema, fix dtypes, handle obvious issues
df_clean, audit = validate_and_clean_data(df)
```

Use `audit` (often a dict or DataFrame-like structure) to quickly see:

* Which columns were renamed or dropped
* What type conversions happened
* How missing values were handled

In a notebook, just display `audit` to inspect it.

---

## 2. Explore the cleaned data

Before you jump into models, get a feel for distributions, missingness, and basic relationships.

```python
from batwing_ml import summary_dataframe, summary_column

summary = summary_dataframe(df_clean)
summary  # nice HTML in notebooks

# Optional: deep dive into one column
age_summary = summary_column(df_clean["age"])
age_summary
```

Typical questions to answer here:

* Are there obvious data quality issues left?
* Are there columns that are clearly junk / IDs / free text?
* Is the target heavily imbalanced?

This is also a good place to manually decide which columns to keep or drop.

---

## 3. Split out features and target

Keep the target separate from the start.

```python
target_col = "label"

X_raw = df_clean.drop(columns=[target_col])
y = df_clean[target_col]
```

You’ll feed `X_raw` into preprocessing and feature engineering next.

---

## 4. Preprocess features

Use Batwing-ML’s preprocessing helper to handle missing values, encoding, scaling, and basic column cleanup.

```python
from batwing_ml import preprocess_dataframe

X_processed, prep_steps = preprocess_dataframe(
    X_raw,
    encode="onehot",      # or "ordinal" depending on your use case
    scale="standard",     # or "minmax" / "robust" / None
    drop_missing_thresh=0.3,
    return_steps=True,
)
```

`prep_steps` will usually contain:

* Which columns were dropped and why (e.g., too many missing values, constant value)
* How numerical and categorical columns were treated
* Any encoding / scaling decisions

This gives you a clean, numeric feature matrix ready for modeling.

---

## 5. Feature engineering

Once preprocessing is done, you can add more signal or reduce noise using `feature_engineering`.

```python
from batwing_ml import feature_engineering

X_fe, fe_meta = feature_engineering(
    X_processed,
    target=None,    # you can also pass the target name if needed by some strategies
    mode="both",   # e.g. "selection", "generation", or "both"
)
```

Depending on your configuration, this step can:

* Drop low-variance or redundant features
* Use model‑based importance or mutual information to select features
* Add synthetic / transformed features
* Optionally apply dimensionality reduction (e.g., PCA)

`fe_meta` gives you a summary of what changed.

Typically at this point you’ll have:

```python
X = X_fe  # final feature matrix
```

---

## 6. Train/validation split

For quick iteration, use a standard train/test split. Nested CV will come later for more rigorous comparisons.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

`stratify=y` is strongly recommended for classification, especially with imbalanced classes.

---

## 7. Hyperparameter tuning for one model (Optuna)

Start with one solid baseline model and tune it with `hyperparameter_tuning_classification`.

```python
from sklearn.ensemble import RandomForestClassifier
from batwing_ml import hyperparameter_tuning_classification

param_grid = {
    "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
    "max_depth":    lambda t: t.suggest_int("max_depth", 3, 20),
    "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10),
}

results = hyperparameter_tuning_classification(
    X=X_train,
    y=y_train,
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    scoring="f1_macro",      # or "roc_auc" for binary, etc.
    fast_mode=False,
)

best_model = results["best_model"]
print(results["best_params"])
print(results["best_score"])
```

Use `fast_mode=True` and/or `use_fraction` / `use_n_samples` when you want very fast feedback on large datasets.

---

## 8. Evaluate the tuned model

Once you have a tuned model, run a richer evaluation on the held‑out test set.

```python
from batwing_ml import evaluate_classification_model

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

This typically gives you:

* Accuracy
* Precision, recall, F1
* ROC AUC (for binary)
* A classification report
* Confusion matrix and optional curves (depending on configuration)

Use this to sanity‑check whether your model is sensible before you get fancy.

---

## 9. Compare multiple models with nested CV (optional but recommended)

When you’re ready to compare models more rigorously, use `run_nested_cv_classification`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from batwing_ml import run_nested_cv_classification

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(),
    "gboost": GradientBoostingClassifier(),
}

param_grids = {
    "logistic": {"C": [0.1, 1, 10]},
    "rf": {"n_estimators": [100, 300]},
    "gboost": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]},
}

nested = run_nested_cv_classification(
    X=X,
    y=y,
    model_dict=models,
    param_grids=param_grids,
    scoring_list=["f1_macro", "roc_auc"],
    search_method="grid",          # or "random"
    return_results=True,
)

nested_summary = nested["summary"]
nested_summary
```

Nested CV helps you answer questions like:

* Which model family is actually best, once tuning is accounted for?
* How stable is performance across folds?
* Are you over‑tuning on one train/test split?

---

## 10. Recommended patterns

A few practical tips when using this workflow:

* **Always keep `X` and `y` separate.** It’s easier to avoid leakage.
* **Log or save `audit`, `prep_steps`, and `fe_meta`.** These make your experiments debuggable and reproducible.
* **Use fast mode early.** On big data, start with `fast_mode=True` + subsampling for quick feedback.
* **Use nested CV when comparing model families.** It’s heavier, but much more honest.

Once this workflow feels comfortable, you can apply the same structure to multiclass classification and regression with the corresponding Batwing-ML functions.
