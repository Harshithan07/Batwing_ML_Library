# Regression Workflow

This workflow mirrors the same clean structure as the classification guide, but tailored for continuous targets. If you're predicting prices, demand, scores, revenue, or anything numeric, this is your go-to recipe.

The flow stays consistent:

**validate → explore → preprocess → engineer → split → tune → evaluate → (optional) nested CV**

---

## 1. Load and validate your data

```python
import pandas as pd
from batwing_ml import validate_and_clean_data

df = pd.read_csv("data.csv")
df_clean, audit = validate_and_clean_data(df)
```

Use `audit` to see type fixes, dropped columns, missing-handling decisions, and schema alignment.

Common early checks for regression:

* Are extreme values real or data errors?
* Is the target column bounded or unbounded?
* Does the distribution look log-normal, skewed, or roughly Gaussian?

---

## 2. Explore your dataset

```python
from batwing_ml import summary_dataframe, summary_column

summary = summary_dataframe(df_clean)
summary
```

Explore the target too:

```python
target_col = "price"

target_summary = summary_column(df_clean[target_col])
target_summary
```

Things to examine:

* Skew/outliers (important for MSE-based metrics)
* Missingness patterns
* Categorical vs numeric balance
* Potential leakage columns

---

## 3. Split features and target

```python
target_col = "price"  # change for your dataset

X_raw = df_clean.drop(columns=[target_col])
y = df_clean[target_col]
```

Keep your target as a separate vector early to avoid accidental leakage in transformation steps.

---

## 4. Preprocess: impute, encode, scale

All standard preprocessing steps for regression stay the same.

```python
from batwing_ml import preprocess_dataframe

X_processed, prep_steps = preprocess_dataframe(
    X_raw,
    encode="onehot",            # often best for regression
    scale="standard",           # ensures fair weighting for linear models
    drop_missing_thresh=0.3,
    return_steps=True,
)
```

What `prep_steps` usually captures:

* Columns removed due to too many missing values
* Categorical columns encoded, numeric columns scaled
* Imputation strategy used

---

## 5. Feature engineering

Feature engineering often matters more in regression than classification.

```python
from batwing_ml import feature_engineering

X_fe, fe_meta = feature_engineering(
    X_processed,
    target=None,
    mode="both",   # transformations + feature selection
)
```

This may:

* Remove irrelevant/low-variance columns
* Add transformed versions of existing features
* Use mutual information or model-based methods to keep strong predictors
* Optionally include PCA or dimensionality reduction if configured

After this step:

```python
X = X_fe
```

---

## 6. Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)
```

For regression, **no stratify needed**.

---

## 7. Hyperparameter tuning for one model (Optuna)

Start with a strong baseline like **RandomForestRegressor**, **XGBoost**, or **GradientBoostingRegressor**.

```python
from sklearn.ensemble import RandomForestRegressor
from batwing_ml import hyperparameter_tuning_regression

param_grid = {
    "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
    "max_depth":    lambda t: t.suggest_int("max_depth", 3, 20),
    "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10),
}

results = hyperparameter_tuning_regression(
    X=X_train,
    y=y_train,
    model_class=RandomForestRegressor,
    param_grid=param_grid,
    scoring="neg_rmse",        # or "neg_mae", "r2" depending on your objective
    fast_mode=False,
)

best_model = results["best_model"]
print(results["best_params"])
print(results["best_score"])
```

Tip: **If your target is heavily skewed**, consider log-transforming `y` before training.

---

## 8. Evaluate the tuned regressor

Use Batwing-ML’s rich evaluation utilities.

```python
from batwing_ml import evaluate_regression_model

metrics = evaluate_regression_model(
    model=best_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    return_dict=True,
)

print(metrics)
```

Typical outputs include:

* R²
* Adjusted R²
* RMSE
* MAE
* MAPE
* RMSLE
* Residual plots
* Error distribution
* Prediction vs Actual scatter

Look for:

* Systematic under/over-prediction
* Heavy tails in residuals
* Patterns indicating missing features

---

## 9. Compare models with nested CV (optional but recommended)

Once you want to compare model families honestly, use nested cross‑validation.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from batwing_ml import run_nested_cv_regression

models = {
    "ridge": Ridge(),
    "lasso": Lasso(),
    "rf": RandomForestRegressor(),
}

param_grids = {
    "ridge": {"alpha": [0.1, 1, 10]},
    "lasso": {"alpha": [0.001, 0.01, 0.1]},
    "rf": {"n_estimators": [100, 300]},
}

nested = run_nested_cv_regression(
    X=X,
    y=y,
    model_dict=models,
    param_grids=param_grids,
    scoring_list=["neg_rmse", "r2", "neg_mae"],
    search_method="grid",
    return_results=True,
)

nested_summary = nested["summary"]
nested_summary
```

Nested CV is extremely helpful when:

* Two models look similar on a train/test split
* You want an unbiased metric for reporting
* You are selecting the final model family for deployment

---

## 10. Recommended regression patterns

A few practical considerations:

### ✔️ Always inspect your target distribution

Transformations matter more here than in classification.

### ✔️ Scale your features

Especially important for linear models and distance-based methods.

### ✔️ Beware of leakage in engineered features

Don’t accidentally encode the target into derived variables.

### ✔️ Tune MAE‑based and RMSE‑based metrics separately

MAE is robust to outliers, RMSE is sensitive — both reveal different truths.

### ✔️ Use nested CV for model family comparison

Especially when deciding between linear, tree‑based, and ensemble methods.

---

Once this workflow feels natural, you can move on to multiclass classification or incorporate the same structure into production‑grade pipelines.
