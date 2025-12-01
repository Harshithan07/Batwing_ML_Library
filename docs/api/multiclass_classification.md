# Multiclass Classification API

This page documents the main entry points for **multiclass** classification workflows.

* Optuna-based hyperparameter tuning for multiclass
* Nested cross-validation
* Multiclass-aware evaluation utilities

> If imports fail during `mkdocs serve`, tweak the module paths in the
> `:::` directives to match your actual package structure.

---

## Hyperparameter tuning (multiclass)

```python
from batwing_ml import hyperparameter_tuning_multiclass_classification
```

::: batwing_ml.hyperparameter_tuning_multiclass_classification

---

## Nested cross-validation (multiclass)

```python
from batwing_ml import run_nested_cv_multiclass_classification
```

::: batwing_ml.run_nested_cv_multiclass_classification

---

## Evaluation utilities (multiclass)

```python
from batwing_ml import evaluate_multiclass_classification
```

::: batwing_ml.evaluate_multiclass_classification
