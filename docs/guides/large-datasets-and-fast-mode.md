# Working with Large Datasets & Fast Mode

Batwing-ML includes several built‑in tools to help you move quickly when handling large datasets. This guide shows you the best practices for fast iteration, safe subsampling, and memory‑efficient experimentation.

This is useful when:

* Your dataset has **hundreds of thousands or millions of rows**
* Hyperparameter tuning is too slow on full data
* Nested cross‑validation takes too long
* You’re running experiments on a laptop or small VM

---

## 1. The philosophy: iterate fast, refine later

For big tabular problems, the most productive workflow is:

1. **Subsample aggressively** (1–20%)
2. **Use fast mode** while tuning
3. **Use simplified model families** early
4. **Scale up only when the pipeline is stable**

This lets you test ideas in seconds or minutes instead of hours.

---

## 2. Subsampling your dataset

You can subsample before or during tuning.

### **Option A — Manual subsample before the pipeline**

```python
df_sampled = df_clean.sample(frac=0.15, random_state=42)
```

Useful when:

* EDA is slow
* Preprocessing itself is heavy

---

### **Option B — Use built‑in subsampling in the tuning utilities**

All hyperparameter tuning functions support:

* `use_fraction`
* `use_n_samples`
* `fast_mode`

Example:

```python
results = hyperparameter_tuning_classification(
    X=X,
    y=y,
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    scoring="f1_macro",
    use_fraction=0.2,   # only 20% used for tuning
)
```

Or:

```python
results = hyperparameter_tuning_regression(
    X=X,
    y=y,
    model_class=RandomForestRegressor,
    param_grid=param_grid,
    use_n_samples=20000,  # cap to 20K rows
)
```

This keeps tuning fast even if the full dataset is huge.

---

## 3. Fast mode during tuning

`fast_mode=True` is your shortcut for quick experimentation.

```python
results = hyperparameter_tuning_classification(
    X=X,
    y=y,
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    fast_mode=True,
)
```

`fast_mode` usually:

* Reduces the number of Optuna trials
* Lowers CV folds
* Disables heavy plots or logs
* Reduces verbosity

This is ideal when you want to test whether:

* The model type is appropriate
* The feature engineering strategy works
* Preprocessing looks correct

Later, you can disable fast mode for the real search.

---

## 4. Fast nested CV

Nested CV can be expensive because:

* Outer folds × inner folds
* Each inner fold runs a full search

For quick comparisons:

```python
nested = run_nested_cv_classification(
    X=X,
    y=y,
    model_dict=models,
    param_grids=param_grids,
    scoring_list=["f1_macro"],
    search_method="random",   # faster than grid
    max_samples=30000,          # limits per-fold data
    fast_mode=True,
    return_results=True,
)
```

### You can control speed with:

* `max_samples` — cap per-fold size
* `fast_mode=True` — reduces folds + search depth
* `search_method="random"` — far lighter than grid

These options let you compare model families in a reasonable time.

---

## 5. Downsampling evaluation

Sometimes even evaluation plots get slow on massive test sets.

Batwing-ML’s evaluation functions let you downsample:

```python
metrics = evaluate_regression_model(
    model=best_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    sample_fraction=0.2,   # only 20% of test used
    return_dict=True,
)
```

Or:

```python
metrics = evaluate_classification_model(
    model=best_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    sample_size=5000,      # use exactly 5000 rows
)
```

This speeds up:

* Confusion matrix
* Residual plots
* ROC/PR curves
* Feature importance calculations

---

## 6. Tips for handling truly large datasets

### ✔️ Use categoricals wisely

High-cardinality features slow down one-hot encoding.

* Drop columns with extreme cardinality
* Switch to `encode="ordinal"` when appropriate

---

### ✔️ Avoid huge dimensionality

One-hot encoding on a million rows × thousands of categories is expensive.

Consider:

* Hashing
* PCA
* Feature selection

---

### ✔️ Start with simpler models

Tree-based models scale better early in experimentation.

Use linear or simpler models when you want very fast iterations.

---

### ✔️ Use small early‑stage Optuna search

```python
fast_mode=True
use_fraction=0.1
```

Then scale up once you’re confident.

---

## 7. Recommended quick‑iteration recipe

For fast discovery:

```python
results = hyperparameter_tuning_classification(
    X=X,
    y=y,
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    fast_mode=True,
    use_fraction=0.1,
)
```

Once your pipeline feels right:

```python
results = hyperparameter_tuning_classification(
    X=X,
    y=y,
    model_class=RandomForestClassifier,
    param_grid=param_grid,
    scoring="f1_macro",
    fast_mode=False,
)
```

---

## 8. Summary

When working with large datasets in Batwing-ML:

* **Subsample early**, especially for EDA and prototyping
* Use **fast_mode** for tuning and nested CV
* Use **random** search methods instead of full grids
* Limit per-fold samples with **max_samples**
* Downsample evaluation with **sample_size** or **sample_fraction**

This lets you stay productive while keeping your experiments honest and repeatable.
