# Binary Classification

End-to-end support for **binary** classification problems: tuning, nested cross-validation, and evaluation.

These functions assume you already have a preprocessed / engineered feature matrix `X` and target `y`.

---

## Hyperparameter tuning

```python
from batwing_ml import hyperparameter_tuning_classification
```

::: batwing_ml.hyperparameter_tuning_classification

---

## Nested cross-validation

```python
from batwing_ml import run_nested_cv_classification
```

::: batwing_ml.run_nested_cv_classification

---

## Evaluation

```python
from batwing_ml import evaluate_classification_model
```

::: batwing_ml.evaluate_classification_model
