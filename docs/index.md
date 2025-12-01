# Batwing Machine Learning Library

Most ML work doesn’t fail because models are hard. It fails because the **data-to-model pipeline is messy, inconsistent, and constantly rewritten** across notebooks, repos, and teams.

Batwing-ML is a lightweight toolkit that fixes that.  
It gives you a **clean, repeatable workflow for tabular ML experiments** — the sort of structure every project needs, but almost nobody has the patience to build from scratch.

Think of it as the missing middle layer between raw pandas code and a full MLOps stack:  
**fast to set up, transparent to debug, and disciplined enough to scale.**

---

## What problem does it solve?

If you work with tabular data long enough, you know the story:

- Every dataset arrives messy.  
- Every experiment needs its own preprocessing.  
- Feature engineering gets copy-pasted from older notebooks.  
- Model tuning turns into a maze of duplicated scripts.  
- Evaluation becomes whatever the last person wrote at 2AM.

Teams end up with **inconsistent pipelines, hidden data leakage, undocumented transformations, ad-hoc hyperparameter searches, and results that can’t be reproduced** a month later.

Batwing-ML gives you a way out by providing:

- **A single, predictable flow** from raw DataFrame → validated features → tuned model → evaluated results.  
- **Structured metadata** that records every transformation and choice.  
- **Notebook-first utilities** for quick EDA, exploration, and diagnostics.  
- **Composable functions** that stay faithful to sklearn and pandas instead of inventing a new ecosystem.  

It’s the workflow you wish you had the first time you shipped a model.

---

## Who is it for?

Batwing-ML is built for:

- **Data scientists** who want fast experiments without drowning in boilerplate.  
- **ML engineers** who want reproducible steps, clean audit trails, and debuggable pipelines.  
- **Students and researchers** who want to run rigorous experiments without reinventing preprocessing or tuning logic.  
- **Teams** who want consistency across notebooks and contributors.

If your work touches **tabular ML**, you’ll probably feel at home.

---

## Why Batwing-ML feels different

Here’s the thing: most “helper” libraries try to abstract everything away.  
Batwing-ML does the opposite — it’s **transparent by design**.

- Every function returns **data + metadata** so you know exactly what happened.  
- Nothing is hidden behind classes or magic fit/transform pipelines.  
- You always keep control of your DataFrame.  
- Everything is powered by standard tools: pandas, numpy, sklearn, Optuna.

You get structure without losing flexibility.

---

## What’s inside

### 1. Data validation  
Catch schema issues, missing columns, broken dtypes, unexpected ranges, and category mismatches — with a clean audit report you can log or inspect.

### 2. EDA & diagnostics  
Quick summaries, feature distributions, correlations, leakage signals, and light model probes.

### 3. Preprocessing  
Imputation, encoding, scaling, deduping, junk-column detection, and clean previews built for notebooks.

### 4. Feature engineering  
Automatic selection (variance, MI, model-based), synthetic features, transformations, PCA, and per-step metadata.

### 5. Hyperparameter tuning  
Optuna-powered tuning across classification, multiclass, and regression — with clean result dictionaries and optional fast-mode sampling.

### 6. Nested cross-validation  
Honest model comparison with minimal code.

### 7. Evaluation  
Consistent metrics, plots, summaries, and export helpers for final model reporting.

Each part works alone, but together they form a **complete experimentation loop**.

---

## A quick end-to-end example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from batwing_ml import (
    validate_and_clean_data,
    preprocess_dataframe,
    feature_engineering,
    hyperparameter_tuning_classification,
)

df = pd.read_csv("data.csv")

# 1) Validate & clean
df_clean, audit = validate_and_clean_data(df)

# 2) Preprocess
X_processed, prep_steps = preprocess_dataframe(
    df_clean.drop(columns=["label"]),
    encode="onehot",
    scale="standard",
    return_steps=True,
)

y = df_clean["label"]

# 3) Engineer features
X_fe, fe_meta = feature_engineering(X_processed)

# 4) Tune a model
results = hyperparameter_tuning_classification(
    X=X_fe,
    y=y,
    model_class=RandomForestClassifier,
    param_grid={
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 300),
        "max_depth":    lambda t: t.suggest_int("max_depth", 3, 20),
    },
    scoring="f1_macro",
)

print(results["best_params"])

```
