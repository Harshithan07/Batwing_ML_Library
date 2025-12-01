import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from tabulate import tabulate
from IPython.display import display, HTML

class _ColumnSteps(dict):
    def _repr_html_(self):
        rows = [f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in self.items()]
        return f"<table style='font-family:monospace;color:#eee;background:#222;padding:8px;'>{''.join(rows)}</table>"
    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.items()])
class _PreprocessingSteps(dict):
    """Custom object to display preprocessing steps nicely in notebooks or as plain text."""

    def _repr_html_(self):
        rows = []
        for key, value in self.items():
            summary = f"{len(value)} items" if isinstance(value, list) and value else "[✔]"
            details = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append((key, summary, details))

        table_html = tabulate(rows, headers=["Step", "Summary", "Details"], tablefmt="unsafehtml")
        return f"<h3>🧠 Preprocessing Steps Summary</h3>{table_html}"

    def __str__(self):
        rows = []
        for key, value in self.items():
            summary = f"{len(value)} items" if isinstance(value, list) and value else "[✔]"
            details = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append((key, summary, details))
        return tabulate(rows, headers=["Step", "Summary", "Details"], tablefmt="fancy_grid")


def _display_scrollable_preview(df: pd.DataFrame, title: str = "Preview"):
    """Internal helper to show a styled, scrollable DataFrame compatible with dark themes like Colab's."""
    html = df.to_html(classes='scroll-table', escape=False, index=False)
    styled_html = f"""
    <style>
        .scroll-table {{
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: monospace;
            font-size: 13px;
            padding: 10px;
            border: 1px solid #444;
            border-radius: 6px;
            max-height: 400px;
            overflow-x: auto;
            overflow-y: auto;
        }}
        .scroll-table table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .scroll-table th {{
            background-color: #2a2a2a;
            color: #ffffff;
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 6px;
            border-bottom: 1px solid #555;
        }}
        .scroll-table td {{
            padding: 6px;
            border-bottom: 1px solid #333;
        }}
        .scroll-table tr:nth-child(even) {{
            background-color: #252525;
        }}
        .scroll-table tr:nth-child(odd) {{
            background-color: #1e1e1e;
        }}
    </style>
    <h3 style='color:#f0f0f0;'>{title}</h3>
    <div class="scroll-table">{html}</div>
    """
    display(HTML(styled_html))



def preprocess_dataframe(
    df: pd.DataFrame,
    impute: bool = True,
    numeric_method: str = 'mean',
    categorical_method: str = 'mode',
    drop_missing_thresh: float = 0.3,
    encode: str = 'onehot',
    scale: str = 'standard',
    drop_constant: bool = True,
    max_cardinality: Optional[int] = 100,
    return_steps: bool = False,
    preview: bool = False,
    verbose: bool = True,
    fast_mode: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Preprocess a tabular dataset for machine learning using a streamlined, transparent pipeline.

    This function performs automatic and configurable preprocessing on a pandas DataFrame, 
    applying common transformations such as missing value imputation, categorical encoding, 
    numeric feature scaling, and filtering of low-information or problematic features.

    It is designed to make raw datasets "model-ready" with minimal manual effort,
    while still allowing for full control over each preprocessing step.

    Parameters
    ----------
    df : pd.DataFrame
        The raw input dataset to preprocess. Must be a standard, rectangular DataFrame.

    impute : bool, default=True
        If True, fills missing values using the specified `numeric_method` and `categorical_method`.

    numeric_method : {'mean', 'median', 'constant'}, default='mean'
        Strategy to fill missing values in numeric columns:
        - 'mean': use column average
        - 'median': use column median
        - 'constant': fill with 0

    categorical_method : {'mode', 'constant', 'drop'}, default='mode'
        Strategy to fill missing values in categorical columns:
        - 'mode': fill with most frequent category
        - 'constant': fill with "missing"
        - 'drop': skip imputation for categoricals

    drop_missing_thresh : float, default=0.3
        Drop columns with more than this proportion of missing values (e.g., 0.3 = 30%).

    encode : {'onehot', 'ordinal', None}, default='onehot'
        How to encode categorical columns:
        - 'onehot': expand categories into binary columns (ideal for tree-based and linear models)
        - 'ordinal': convert categories into integers (compact but model-sensitive)
        - None: skip encoding

    scale : {'standard', 'minmax', 'robust', None}, default='standard'
        Scaling method for numeric features:
        - 'standard': zero mean, unit variance (default for most ML models)
        - 'minmax': scales features to [0, 1]
        - 'robust': scales using median and IQR (more resistant to outliers)
        - None: skip scaling

    drop_constant : bool, default=True
        If True, drops columns where all values are the same — these provide no predictive value.

    max_cardinality : int or None, default=100
        If set, drops categorical columns with more than this many unique values.
        Useful to eliminate IDs or high-entropy features that aren't generalizable.

    return_steps : bool, default=False
        If True, returns a second object (`steps`) summarizing the preprocessing pipeline steps.
        This is useful for auditing, documentation, or reproducing pipelines.

    preview : bool, default=False
        If True, displays a styled scrollable HTML preview of the top rows of the transformed DataFrame.
        Intended for use inside notebooks (e.g., Jupyter, Colab).

    verbose : bool, default=True
        If True, prints human-readable logs describing each transformation as it is applied.

    fast_mode : bool, default=False
        If True, disables logging and previews to optimize performance for large datasets (1M+ rows).
        Recommended for production or batch processing pipelines.

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, dict]
        - The transformed DataFrame, ready for use in ML models.
        - If `return_steps=True`, also returns a dictionary-like object listing:
            - dropped columns (due to high missingness, constant value, or high cardinality)
            - columns encoded, scaled, or imputed
            - final column names

    Examples
    --------
    >>> df_clean = preprocess_dataframe(df)

    >>> df_clean, steps = preprocess_dataframe(
            df,
            encode="ordinal",
            scale="robust",
            drop_missing_thresh=0.25,
            return_steps=True,
            preview=True
        )
    >>> print(steps)

    Notes
    -----
    - This function does not modify the input DataFrame (works on a copy).
    - Designed to be flexible enough for prototyping, reproducible for experiments,
      and fast enough for production workloads.
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - For column-wise control, see `preprocess_column()`.

    See Also
    --------
    preprocess_column : Clean and transform a single Series with similar options.

    User Guide
    ----------
    🧭 When Should You Use This Function?
    - You're starting with raw, messy tabular data that has missing values, mixed data types, or irrelevant columns.
    - You want to transform the dataset into a clean, numerical form ready for modeling — without hardcoding pipelines manually.
    - You're preparing data for machine learning algorithms (scikit-learn, XGBoost, LightGBM, etc.) and need a reproducible cleaning strategy.
    - You're in an experimentation phase and want fast iteration with logs, or you're moving toward deployment and need performance mode.

    ⚙️ Recommended Configurations (Use-Case Based)

    1. **General-purpose ML modeling (balanced tabular data):**
       → Works well with logistic regression, SVMs, and shallow neural networks.
       >>> preprocess_dataframe(df, encode="onehot", scale="standard")

       *Why?* One-hot encoding ensures categorical variables are treated independently. Standard scaling helps models converge.

    2. **Tree-based models (RandomForest, XGBoost, LightGBM):**
       → These models handle ordinal input well and don’t require scaling.
       >>> preprocess_dataframe(df, encode="ordinal", scale=None)

       *Why?* One-hot can add noise or bloat tree-based models. Ordinal + no scaling is faster and sufficient.

    3. **High-cardinality datasets or sparse data (e.g., recommender systems):**
       >>> preprocess_dataframe(df, encode=None, max_cardinality=50)

       *Why?* Skip encoding and limit high-cardinality columns to avoid exploding the feature space.

    4. **Production or big data batch preprocessing:**
       → Great for pipelines where performance > visuals.
       >>> preprocess_dataframe(df, fast_mode=True)

       *Why?* Disables all logging and display overhead — ideal for 1M+ rows.

    5. **Auditing or debugging preprocessing behavior:**
       → You want to know exactly what was changed and why.
       >>> df_clean, steps = preprocess_dataframe(df, return_steps=True, verbose=True)
       >>> print(steps)

       *Why?* `steps` logs dropped columns, encoded features, and final output — great for versioning and reproducibility.


    💡 Tips:
    - Use `preview=True` only in notebooks to visualize output cleanly.
    - Set `max_cardinality=None` to retain all categorical columns, even high-card ones.
    - `drop_constant=True` removes junk features automatically.
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - Use `preprocess_column()` for detailed tuning on a single feature.

    Related:
    --------
    • preprocess_column(): Clean and transform a single Series with similar options.
    • summary_column(): View deep stats for a single column before preprocessing
    • summary_dataframe(): View deep stats for a DataFrame before preprocessing
    
    """

    df = df.copy()
    steps = {}

    # Fast mode: heavy preprocessing skipped
    if fast_mode:
        preview = False
        verbose = False

    # Drop high-missing columns
    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > drop_missing_thresh].index.tolist()
    if high_missing and verbose:
        print(f"[Drop] {len(high_missing)} columns dropped for >{int(drop_missing_thresh * 100)}% missing.")
    df.drop(columns=high_missing, inplace=True)
    steps['dropped_high_missing'] = high_missing

    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    if drop_constant and constant_cols:
        if verbose:
            print(f"[Drop] {len(constant_cols)} constant columns dropped.")
        df.drop(columns=constant_cols, inplace=True)
    steps['dropped_constant'] = constant_cols

    # Drop high-cardinality categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    high_card_cols = [col for col in cat_cols if df[col].nunique() > max_cardinality] if max_cardinality else []
    if high_card_cols:
        if verbose:
            print(f"[Drop] {len(high_card_cols)} high-cardinality columns dropped.")
        df.drop(columns=high_card_cols, inplace=True)
    steps['dropped_high_cardinality'] = high_card_cols

    # Refresh column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = []

    # Imputation
    if impute:
        if num_cols:
            transformers.append(('imputer_num', SimpleImputer(strategy=numeric_method), num_cols))
            if verbose:
                print(f"[Impute] Numeric → {numeric_method} on {len(num_cols)} cols")
        if cat_cols and categorical_method != 'drop':
            strategy = 'most_frequent' if categorical_method == 'mode' else 'constant'
            transformers.append(('imputer_cat', SimpleImputer(strategy=strategy), cat_cols))
            if verbose:
                print(f"[Impute] Categorical → {categorical_method} on {len(cat_cols)} cols")
    elif verbose:
        print("[Impute] Skipped")

    # Encoding
    if encode and cat_cols:
        if encode == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif encode == 'ordinal':
            encoder = OrdinalEncoder()
        else:
            raise ValueError("Invalid encoding method")
        transformers.append(('encoder', encoder, cat_cols))
        if verbose:
            print(f"[Encode] {encode} encoding applied")
    elif verbose:
        print("[Encode] Skipped")

    # Scaling
    if scale and num_cols:
        if scale == 'standard':
            scaler = StandardScaler()
        elif scale == 'minmax':
            scaler = MinMaxScaler()
        elif scale == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scale method")
        transformers.append(('scaler', scaler, num_cols))
        if verbose:
            print(f"[Scale] {scale} scaling applied")
    elif verbose:
        print("[Scale] Skipped")

    # Apply ColumnTransformer
    if transformers:
        pipeline = ColumnTransformer(transformers, remainder='drop')
        df_transformed = pipeline.fit_transform(df)

        # Generate readable column names
        feature_names = []
        for name, transformer, cols in pipeline.transformers_:
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(cols)
                except:
                    names = [f"{name}_{col}" for col in cols]
            else:
                names = [f"{col}" for col in cols]
            feature_names.extend(names)

        df = pd.DataFrame(df_transformed, columns=feature_names)
        steps['final_columns'] = feature_names

        if verbose:
            print(f"[Transform] Final shape: {df.shape}")
    elif verbose:
        print("[Transform] No transformations applied")

    if preview:
        _display_scrollable_preview(df.head(10), title="🧼 Preprocessed DataFrame (Top 10 Rows)")

    return (df, _PreprocessingSteps(steps)) if return_steps else df
