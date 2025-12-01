# Batwing ML Library

**Modular, functional, and interpretable machine learning pipeline** for classification, multiclass, and regression tasks. Designed for data scientists who want rapid experimentation, clean diagnostics, and powerful model comparisons — all with minimal code.

---

## 🚀 Features

* Full EDA and column-level diagnosis
* Modular preprocessing (impute, encode, scale)
* Feature engineering with PCA and correlation filtering
* Hyperparameter tuning with Optuna (classification, regression, multiclass)
* Nested Cross-Validation for robust model benchmarking
* Rich model evaluation (metrics + visualizations)
* Supports cost-sensitive classification and diagnostics
* Dashboard/notebook-friendly outputs (HTML/tabulate/rich)

---

## 📦 Installation

> Coming soon to PyPI

For now, clone the repo and import functions directly:

```bash
git clone https://github.com/your-org/batwing-ml.git
```

```python
from batwing_ml import (
    summary_dataframe,
    preprocess_dataframe,
    run_nested_cv_classification,
    evaluate_classification_model,
    ...
)
```

---

## 🧠 Module Overview

| Module                                               | Key Functions                                                                 |
| ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| `exploratory.py`                                     | `summary_dataframe()`, `summary_column()` – full EDA, missing patterns, plots |
| `data_validation_and_etl.py`                         | Data shape, type, duplication checks                                          |
| `data_preparation.py`                                | Label transformation, type casting, etc.                                      |
| `feature_engineering.py`                             | PCA, correlation pruning, importance plots                                    |
| `preprocessor.py`                                    | `preprocess_dataframe()`, `preprocess_column()` – encode, scale, impute       |
| `hyperparameter_tuning_classification.py`            | Optuna tuning for binary classification                                       |
| `run_nested_cv_classification.py`                    | Nested CV with model benchmarking                                             |
| `evaluate_classification_model.py`                   | Confusion matrix, ROC, cost-sensitive plots                                   |
| `hyperparameter_tuning_multiclass_classification.py` | Multiclass Optuna tuning                                                      |
| `nested_cv_multiclass_classification.py`             | Nested CV for multiclass tasks                                                |
| `evaluate_multiclass_classification.py`              | Precision, recall, per-class analysis                                         |
| `hyperparameter_tuning_regression.py`                | Optuna tuning for regression                                                  |
| `nested_cv_regression.py`                            | Nested CV for regression models                                               |
| `evaluate_regression_model.py`                       | Regression metrics + diagnostic plots                                         |

---

## 🔧 Usage Examples

### 📊 1. Data Summary

```python
summary_dataframe(df, verbose=True, detailing=True, correlation_matrix=True)
summary_column(df, "age", plots=["histogram", "missing_trend"])
```

### ⚙️ 2. Preprocessing

```python
X_proc, y_proc, steps = preprocess_dataframe(
    df, target_col="target",
    impute=True, encode="onehot", scale="standard",
    return_steps=True
)
```

### 🔁 3. Model Tuning (Binary Classification)

```python
from sklearn.ensemble import RandomForestClassifier
from batwing_ml import hyperparameter_tuning_classification

model_class = RandomForestClassifier
param_grid = {
    'n_estimators': lambda trial: trial.suggest_int("n_estimators", 50, 200),
    'max_depth': lambda trial: trial.suggest_int("max_depth", 3, 10)
}

results = hyperparameter_tuning_classification(
    model_class=model_class,
    param_grid=param_grid,
    X=X, y=y,
    scoring='roc_auc'
)
```

### 🏁 4. Nested Cross-Validation (Regression)

```python
models = {
    "ridge": Ridge(),
    "rf": RandomForestRegressor()
}
param_grids = {
    "ridge": {"alpha": [0.1, 1.0, 10]},
    "rf": {"n_estimators": [100, 200], "max_depth": [3, 5]}
}

run_nested_cv_regression(
    X=X, y=y,
    model_dict=models,
    param_grids=param_grids,
    scoring_list=["r2", "rmse", "mae"],
    search_method="grid",
    return_results=True
)
```

---

## 📈 Visualizations

* Feature Importance
* Correlation Heatmaps
* PCA Scree and Scatter Plots
* Confusion Matrix, ROC, Threshold Plots
* Learning Curve, Residuals, Prediction vs Actual
* Lift Charts, Cost-Sensitive Curves

---

## 📚 License

MIT License

---

## 👥 Contributors

Built by \[Your Name] and contributors.

---

## 💡 Future Additions

* AutoML wrappers
* MLflow integration
* HTML dashboard export
* Time series module

---
