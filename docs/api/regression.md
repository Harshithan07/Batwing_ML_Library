# Regression API

This page documents the regression-side entry points in Batwing-ML.

* Optuna-based hyperparameter tuning
* Nested cross-validation for regression models
* Rich regression evaluation and diagnostics

> As before, if mkdocstrings complains about an import, adjust the
> dotted module paths below.

---

## Hyperparameter tuning (regression)

```python
from batwing_ml import hyperparameter_tuning_regression
```

::: batwing_ml.hyperparameter_tuning_regression

---

## Nested cross-validation (regression)

```python
from batwing_ml import run_nested_cv_regression
```

::: batwing_ml.run_nested_cv_regression

---

## Evaluation utilities (regression)

```python
from batwing_ml import evaluate_regression_model
```

::: batwing_ml.evaluate_regression_model
