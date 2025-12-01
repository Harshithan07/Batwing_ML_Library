# Preprocessing

Turn a cleaned DataFrame into a modeling-ready feature matrix.

This step handles:

* Missing values
* Encoding categoricals
* Scaling numeric features
* Dropping junk / constant / high-missing columns

```python
from batwing_ml import preprocess_dataframe, preprocess_column
```

## DataFrame-level preprocessing

::: batwing_ml.preprocess_dataframe

## Column-level preprocessing

::: batwing_ml.preprocess_column
