# Exploratory & ETL API

This section documents the core utilities for data validation, sampling/splitting, EDA summaries, preprocessing, and feature engineering.

> Note: The `:::` blocks below are mkdocstrings hooks. If your module paths differ slightly,
> adjust the dotted paths (e.g. `batwing_ml.something.module`) to match your actual package.

---

## Data validation & ETL

High-level entry point for cleaning and validating raw tabular data.

```python
from batwing_ml import validate_and_clean_data
```

::: batwing_ml.validate_and_clean_data

---

## Data preparation & splitting

Helpers for sampling and preparing train/val/test splits.

```python
from batwing_ml import prepare_sample_split
```

::: batwing_ml.prepare_sample_split

---

## DataFrame & column summaries

Notebook-friendly EDA summaries for full DataFrames and individual columns.

```python
from batwing_ml import summary_dataframe, summary_column
```

::: batwing_ml.summary_dataframe

::: batwing_ml.summary_column

---

## Preprocessing helpers

High-level preprocessing utilities that handle imputation, encoding, scaling, and column cleanup.

```python
from batwing_ml import preprocess_dataframe, preprocess_column
```

::: batwing_ml.preprocess_dataframe

::: batwing_ml.preprocess_column

---

## Feature exploration

Lightweight utilities to inspect feature-target relationships.

```python
from batwing_ml import feature_exploration
```

::: batwing_ml.feature_exploration

---

## Feature engineering

Tools for feature selection, transformation, and dimensionality reduction.

```python
from batwing_ml import feature_engineering
```

::: batwing_ml.feature_engineering
