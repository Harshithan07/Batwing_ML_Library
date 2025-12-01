import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional, Tuple
from scipy.stats import zscore, iqr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from IPython.display import display, HTML
import warnings

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

def preprocess_column(
    col: pd.Series,
    impute: Optional[Union[str, float, int]] = None,
    scale: Optional[str] = None,
    encode: Optional[str] = None,
    cap_outliers: Optional[str] = None,
    cap_quantiles: Tuple[float, float] = (0.01, 0.99),
    custom_map: Optional[Callable] = None,
    strategy: str = "manual",
    inplace: bool = False,
    preview: bool = False,
    plot: bool = False,
    verbose: bool = True,
    fast_mode = False
) -> Union[pd.Series, Tuple[pd.Series, dict]]:
    """
    Preprocess a single column of data with smart transformations for modeling.

    This function allows targeted preprocessing of a single pandas Series — such as a feature column
    from a DataFrame — including missing value imputation, outlier treatment, encoding, scaling, 
    custom transformations, and optional inspection via plots or previews.

    It is useful for exploratory analysis, fine-tuned feature engineering, or inspecting
    column-specific cleaning steps outside of full-pipeline automation.

    Parameters
    ----------
    col : pd.Series
        The input column to clean. Can be numeric or categorical.

    impute : {'mean', 'median', 'mode', 'constant'} or scalar, optional
        Strategy to fill missing values:
        - 'mean', 'median': numeric only
        - 'mode': most frequent value (works for both types)
        - 'constant': fill with 0 or "missing"
        - scalar: fill with a specific value (e.g., 0 or 'unknown')

    scale : {'standard', 'minmax', 'robust'}, optional
        If provided, applies scaling (numeric only):
        - 'standard': zero mean, unit variance
        - 'minmax': rescale to [0, 1]
        - 'robust': scale based on median and IQR (for outliers)

    encode : {'ordinal', 'onehot'}, optional
        If provided, applies encoding (categorical only):
        - 'ordinal': converts categories to integer codes
        - 'onehot': returns a new DataFrame with binary indicator columns

    cap_outliers : {'zscore', 'iqr'}, optional
        Method for outlier detection and clipping (numeric only):
        - 'zscore': clips values with |Z| > 3
        - 'iqr': clips values outside 1.5 * IQR from Q1/Q3

    cap_quantiles : tuple of float, default=(0.01, 0.99)
        Lower and upper quantiles to cap values (a form of Winsorization).
        Applied only for numeric columns.

    custom_map : callable, optional
        A custom function applied to every non-null element (e.g., `np.log1p`, `str.lower`).
        Useful for transformations like scaling, mapping, or normalization.

    strategy : {'manual', 'auto'}, default='manual'
        - 'manual': use the specified arguments only
        - 'auto': infer reasonable defaults based on column dtype and missing values

    inplace : bool, default=False
        If True, modifies the original Series inside a DataFrame. Otherwise works on a copy.

    preview : bool, default=False
        If True, prints a small sample (head) of the cleaned column.

    plot : bool, default=False
        If True, shows a histogram or bar plot (before/after if possible) to visualize the distribution.

    verbose : bool, default=True
        If True, logs steps taken (e.g., imputed with mean, encoded with ordinal).

    fast_mode : bool, default=False
        If True, disables plotting and previews for faster execution on large data.

    Returns
    -------
    pd.Series or Tuple[pd.Series, dict]
        - Cleaned Series (or one-hot encoded DataFrame if applicable).
        - If used in unpacking (`col_clean, steps = ...`), also returns a dictionary
          detailing the preprocessing actions taken.

    Examples
    --------
    >>> col_clean = preprocess_column(df["age"], impute="mean", scale="standard")[0]
    
    >>> cat_col, steps = preprocess_column(
            df["gender"],
            impute="mode",
            encode="ordinal",
            preview=True,
            verbose=True
        )
    >>> print(steps)

    User Guide
    ----------
    🧭 When Should You Use This?
    - You want fine-grained control over a **single column** in your dataset — e.g., apply different transformations for different features.
    - You’re doing **exploratory data analysis** and want to inspect the impact of transformations before applying them in batch.
    - You’re manually curating a feature set for a model and want to test encoding, scaling, or outlier treatment interactively.
    - You’re building a **custom preprocessing function** per column for pipelines.

    ⚙️ Recommended Workflows (Based on Column Type)

    1. **For numeric columns (e.g., age, income, price):**
       Apply standard cleaning + scaling + clipping:
       >>> col_clean, steps = preprocess_column(
               df["income"],
               impute="mean",
               scale="standard",
               cap_outliers="zscore",
               cap_quantiles=(0.01, 0.99),
               preview=True,
               plot=True
           )

       *Why?* Numeric data benefits from scaling and outlier handling for better model convergence.

    2. **For categorical columns (e.g., gender, city):**
       Apply imputation and ordinal encoding:
       >>> col_clean = preprocess_column(
               df["gender"],
               impute="mode",
               encode="ordinal"
           )[0]

       *Why?* Many models expect numeric input; ordinal encoding works well for tree models.

    3. **When exploring transformation effects:**
       Use custom mapping functions or log transforms:
       >>> preprocess_column(df["price"], custom_map=np.log1p, plot=True)

       *Why?* Helps reduce skew in price-like data, which improves linear model performance.

    4. **Quick automation:**
       Let the function auto-decide how to handle the column:
       >>> preprocess_column(df["feature_x"], strategy="auto")

       *Why?* Saves time when you’re cleaning many features quickly.

    🔍 Tips:
    - Use `preview=True` to view how the column looks after cleaning.
    - Use `plot=True` to visualize distributions before/after transformations.
    - Use `steps` output to document what happened to the column — especially useful in notebooks or reports.
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - One-hot encoding returns a DataFrame (not Series) — plan how you store or merge it back.

    Related:
    --------
    • preprocess_dataframe(): Clean an entire DataFrame in one call
    • summary_column(): View deep stats for a single column before preprocessing
    • summary_dataframe(): View deep stats for a DataFrame before preprocessing

    """
    col = col.copy()
    original_dtype = col.dtype
    steps = _ColumnSteps()
    steps["original_dtype"] = str(original_dtype)

    # Fastmode
    if fast_mode:
      preview = False
      verbose = False
      plot = False
    
    # Strategy
    if strategy == "auto":
        if pd.api.types.is_numeric_dtype(col):
            impute = impute or "mean"
            scale = scale or "standard"
        elif pd.api.types.is_categorical_dtype(col) or col.dtype == object:
            impute = impute or "mode"
            encode = encode or "ordinal"

    # Imputation
    if impute is not None:
        if impute == "mean":
            fill_value = col.mean()
        elif impute == "median":
            fill_value = col.median()
        elif impute == "mode":
            fill_value = col.mode().iloc[0] if not col.mode().empty else None
        elif impute == "constant":
            fill_value = 0
        elif isinstance(impute, (int, float, str)):
            fill_value = impute
        else:
            raise ValueError("Invalid impute method")
        col.fillna(fill_value, inplace=True)
        steps["imputed"] = f"{impute} → {fill_value}"

    # Custom map
    if custom_map:
        col = col.apply(lambda x: custom_map(x) if pd.notnull(x) else x)
        steps["custom_map"] = custom_map.__name__ if hasattr(custom_map, '__name__') else 'lambda'

    # Outlier Capping
    if cap_outliers == "zscore":
        zs = zscore(col.dropna())
        mask = np.abs(zs) > 3
        capped_vals = col[~mask]
        q1, q2 = capped_vals.min(), capped_vals.max()
        col = np.clip(col, q1, q2)
        steps["outlier_cap"] = f"zscore > 3 → clipped to [{q1:.2f}, {q2:.2f}]"

    elif cap_outliers == "iqr":
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr_val = q3 - q1
        lower = q1 - 1.5 * iqr_val
        upper = q3 + 1.5 * iqr_val
        col = np.clip(col, lower, upper)
        steps["outlier_cap"] = f"IQR → clipped to [{lower:.2f}, {upper:.2f}]"

    # Capping (Winsorization)
    if cap_quantiles and pd.api.types.is_numeric_dtype(col):
        low, high = col.quantile(cap_quantiles[0]), col.quantile(cap_quantiles[1])
        col = np.clip(col, low, high)
        steps["winsorize"] = f"Capped at {cap_quantiles[0]*100:.0f}–{cap_quantiles[1]*100:.0f}% quantiles"
    elif cap_quantiles and not pd.api.types.is_numeric_dtype(col):
      if verbose:
        warnings.warn("Winsorization skipped: column is not numeric.")

    # Encoding (Categorical)
    if encode and (col.dtype == "object" or col.dtype.name == "category"):
        if encode == "ordinal":
            col = pd.Series(OrdinalEncoder().fit_transform(col.values.reshape(-1, 1)).flatten(), index=col.index)
            steps["encoded"] = "ordinal"
        elif encode == "onehot":
            encoded_df = pd.get_dummies(col, prefix=col.name)
            steps["encoded"] = f"onehot → {len(encoded_df.columns)} cols"
            return encoded_df, steps
        else:
            raise ValueError("Invalid encoding method")

    # Scaling
    if scale and pd.api.types.is_numeric_dtype(col):
        scaler = {"standard": StandardScaler(),
                  "minmax": MinMaxScaler(),
                  "robust": RobustScaler()}.get(scale)

        if scaler:
            col_vals = col.values.reshape(-1, 1)
            scaled = scaler.fit_transform(col_vals).flatten()
            col = pd.Series(scaled, index=col.index)
            steps["scaled"] = scale
        else:
            raise ValueError("Invalid scale method")

    if preview:
        display(HTML(col.head(10).to_frame().to_html()))
    if plot:
        plt.figure(figsize=(8, 3))
        if pd.api.types.is_numeric_dtype(col):
            plt.hist(col.dropna(), bins=30, color="#66b3ff", edgecolor="k")
            plt.title("Numeric Column Distribution")
        else:
            col.value_counts().head(20).plot(kind="bar", color="#66b3ff", edgecolor="k")
            plt.title("Categorical Column Frequency")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    steps["final_dtype"] = str(col.dtype)
    return col, steps