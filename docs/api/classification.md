# Binary Classification API

This page documents the main entry points for binary classification experiments in Batwing-ML.

* Optuna-based hyperparameter tuning
* Nested cross-validation for unbiased model comparison
* Rich evaluation utilities (metrics + plots)

> If mkdocstrings fails to import something, adjust the dotted paths below
> to match your actual module layout.

---

## Hyperparameter tuning (binary classification)

```python
from batwing_ml import hyperparameter_tuning_classification
```

::: batwing_ml.hyperparameter_tuning_classification

---

## Nested cross-validation (binary classification)

```python
from batwing_ml import run_nested_cv_classification
```

::: batwing_ml.run_nested_cv_classification

---

## Evaluation utilities (binary classification)

```python
from batwing_ml import evaluate_classification_model
```

::: batwing_ml.evaluate_classification_model
    