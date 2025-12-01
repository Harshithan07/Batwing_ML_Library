import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew, entropy
from tabulate import tabulate
from IPython.display import display, HTML

def summary_dataframe(df: pd.DataFrame, verbose: bool = True, return_dataframes: bool = False,
                      detailing: bool = False, correlation_matrix: bool = False, fast_mode: bool = False):
    """
    Generates a detailed summary report of an entire DataFrame for exploratory data analysis (EDA).

    This function provides a transparent, column-by-column overview of your dataset, including
    data types, missing value patterns, uniqueness, cardinality, and optionally deeper
    statistical insights like skewness, kurtosis, entropy, and correlation structure.

    It helps you understand the shape and quality of your data before modeling, and supports
    structured auditing via optional DataFrame outputs for downstream usage.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to analyze. Must be a rectangular DataFrame with rows and columns.

    verbose : bool, default=True
        If True, prints human-readable summaries using styled tables, markdown, or HTML (in notebooks).
        If `fast_mode=True`, this will be ignored (no prints will be shown).

    return_dataframes : bool, default=False
        If True, returns the core summary tables as pandas DataFrames for programmatic use.

    detailing : bool, default=False
        If True, enables additional statistics:
        - For numeric: skewness, kurtosis, and z-score-based outlier counts
        - For categorical: entropy (how evenly distributed the values are)
        - Also shows duplicate row and column analysis

    correlation_matrix : bool, default=False
        If True, computes a Pearson correlation matrix for all numeric features.

    fast_mode : bool, default=False
        If True, disables all visual and computationally expensive diagnostics:
        - Skips entropy, skewness, kurtosis, outliers, and duplicate detection
        - Skips correlation matrix computation
        - Skips verbose display/logging
        Use this mode when profiling large datasets (e.g., 1M+ rows) or in batch workflows.

    Returns
    -------
    tuple of pd.DataFrame, optional
        Only returned if `return_dataframes=True`. Includes:
        - summary : Core metadata (dtype, missing %, unique %, etc.)
        - desc_numeric : Descriptive stats for all numeric columns
        - desc_categorical : Descriptive stats for categorical/object columns
        - correlation_matrix : Numeric correlation matrix (only if `correlation_matrix=True`)

    Raises
    ------
    ValueError
        If the input DataFrame is empty or not valid.

    Examples
    --------
    >>> summary_dataframe(df, detailing=True, correlation_matrix=True)

    >>> summary, num_stats, cat_stats = summary_dataframe(df, return_dataframes=True, detailing=True)

    Notes
    -----
    - The function is non-destructive: it reads from the input DataFrame without modifying it.
    - If you're working with extremely large datasets, set `fast_mode=True` to avoid slow diagnostics.
    - When `verbose=True`, this function uses IPython’s HTML renderer for a notebook-friendly display.

    See Also
    --------
    summary_column : Analyze a single column with detailed metrics and plots
    preprocess_dataframe : Prepare a dataset for modeling through scaling, encoding, and imputation
    preprocess_column : Clean a single column manually (e.g., outlier handling, transformation)

    User Guide
    ----------
    🧠 When Should You Use This?
    - At the start of a project to assess **data readiness**.
    - Before feature engineering to identify **columns to drop, fix, or transform**.
    - During EDA or notebook exploration to communicate **data quality**.
    - In automated pipelines where you need **programmatic summary outputs**.

    📌 What You'll Learn:
    - Which columns have high missingness, low variance, or high cardinality
    - How many numeric/categorical features exist
    - Skewness or entropy in features (if detailing=True)
    - Whether your dataset has duplicated rows or columns
    - Correlation patterns among numeric features (optional)

    ⚙️ Recommended Usage Patterns:

    1. **Full EDA diagnostic (notebooks):**
       >>> summary_dataframe(df, detailing=True, correlation_matrix=True)

    2. **For dashboards or programmatic reporting:**
       >>> summary, num_stats, cat_stats = summary_dataframe(df, return_dataframes=True)

    3. **Batch analysis or large files:**
       >>> summary_dataframe(df, fast_mode=True)

    4. **Minimal quick check (CLI or scripts):**
       >>> summary_dataframe(df, detailing=False, verbose=True)

    💡 Tips:
    - Use with `preprocess_dataframe()` to act on low-quality features you identify here.
    - `entropy` close to 0 → one category dominates (low information)
    - High skew/kurtosis → consider log or robust transformations
    - Z-score outliers >10 → column likely needs clipping or scaling
    - Use `fast_mode=True` when processing high-volume datasets or in production loops.
    - Use `detailing=False` for a quick overview of the dataset without deep stats.
    - Use `correlation_matrix=False` to skip the correlation matrix.
    - Use `return_dataframes=True` to export summaries to reports or ML audit logs
    """

    if fast_mode:
        detailing = False
        correlation_matrix = False
        verbose = False

    if df.empty:
        raise ValueError("The provided DataFrame is empty. Provide a valid dataset.")

    total_rows = df.shape[0]
    numeric_df = df.select_dtypes(include=["number"])
    categorical_df = df.select_dtypes(include=["object", "category"])

    summary = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Total Values": df.count().values,
        "Missing Values": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / total_rows * 100).round(2),
        "Unique Values": df.nunique().values,
        "Unique %": (df.nunique().values / total_rows * 100).round(2)
    })

    summary["Constant Column"] = summary["Unique Values"] == 1
    summary["Cardinality Category"] = summary["Unique Values"].apply(
        lambda x: "Low" if x <= 10 else "Medium" if x <= 100 else "High"
    )

    if detailing:
        duplicate_rows = df.duplicated().sum()
        try:
            duplicate_columns = df.T.duplicated().sum()
        except Exception:
            duplicate_columns = "Too large to compute"

        if not numeric_df.empty:
            desc_numeric = numeric_df.describe().transpose()
            desc_numeric["Skewness"] = numeric_df.apply(lambda x: skew(x.dropna()), axis=0)
            desc_numeric["Kurtosis"] = numeric_df.apply(lambda x: kurtosis(x.dropna()), axis=0)
            desc_numeric["Z-score Outliers"] = numeric_df.apply(
                lambda x: (np.abs((x - x.mean()) / x.std()) > 3).sum(), axis=0
            )
        else:
            desc_numeric = None

        if not categorical_df.empty:
            desc_categorical = categorical_df.describe().transpose()
            desc_categorical["Entropy"] = categorical_df.apply(
                lambda x: entropy(x.value_counts(normalize=True), base=2) if x.nunique() > 1 else 0
            )
        else:
            desc_categorical = None
    else:
        duplicate_rows = None
        duplicate_columns = None
        desc_numeric = numeric_df.describe().transpose() if not numeric_df.empty else None
        desc_categorical = categorical_df.describe().transpose() if not categorical_df.empty else None

    corr_matrix = numeric_df.corr() if (not numeric_df.empty and correlation_matrix) else None

    if verbose:
        def show_df(df_, title):
            if df_ is not None:
                html = df_.to_html(classes='scroll-table', escape=False)
                display(HTML(f"<h3>{title}</h3>" + html))

        display(HTML("""
        <style>
        .scroll-table {
            display: block;
            max-height: 400px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 14px;
        }
        </style>
        """))

        show_df(summary, "Summary Statistics")
        show_df(desc_numeric, "Descriptive Statistics (Numerical)")
        show_df(desc_categorical, "Descriptive Statistics (Categorical)")
        show_df(corr_matrix, "Correlation Matrix")

        if detailing:
            print(f"\nTotal Duplicate Rows: {duplicate_rows}")
            print(f"Total Duplicate Columns: {duplicate_columns}")

    if return_dataframes:
        return summary, desc_numeric, desc_categorical, corr_matrix
