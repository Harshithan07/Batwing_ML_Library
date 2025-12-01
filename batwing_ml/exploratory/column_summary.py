import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew, entropy
from tabulate import tabulate
from IPython.display import display, HTML

def summary_column(df: pd.DataFrame, column_name: str, top_n: int = 10,
                   verbose: bool = True, return_dataframes: bool = False,
                   detailing: bool = True, time_column: str = None,
                   plots: list = None, fast_mode: bool = False):
    """
    Generates a detailed, human-readable summary of a single column in a DataFrame.

    This function helps you understand the nature of a specific column by providing
    summary metrics such as missingness, uniqueness, cardinality, entropy, skewness,
    and optional visualizations. It adapts intelligently to both numeric and categorical data
    and can highlight distribution issues, outliers, or missing trends over time.

    It is ideal for exploratory data analysis (EDA), column-wise diagnostics, and
    auditing feature quality before preprocessing or modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column to summarize.

    column_name : str
        The name of the column to analyze.

    top_n : int, default=10
        Number of most frequent values to display in the frequency table for categorical features.

    verbose : bool, default=True
        If True, prints all summaries in formatted tables with headers. If `fast_mode=True`, this is ignored.

    return_dataframes : bool, default=False
        If True, returns:
        - summary_table : key metrics (missing %, unique %, etc.)
        - desc_stats : descriptive stats (mean, std, IQR, etc.)
        - freq_dist : frequency counts of top values

    detailing : bool, default=True
        If True, enables additional diagnostics:
        - For numeric: skewness, kurtosis, z-score outliers
        - For categorical: entropy
        - Also enables visualizations if `plots` is specified

    time_column : str, optional
        If specified, enables a missing-value trend chart over time using this datetime column.
        Useful for temporal datasets and time-series analysis.

    plots : list of str, optional
        List of plots to display:
        - "histogram": for numeric distribution
        - "bar": for top category counts
        - "missing_trend": for missing rate over time (requires `time_column`)

    fast_mode : bool, default=False
        If True, skips all optional visualizations, skew/entropy calculations, and pretty print tables.
        Recommended when analyzing very large datasets or when integrating into production pipelines.

    Returns
    -------
    tuple of pd.DataFrame, optional
        If `return_dataframes=True`, returns:
        - summary_table : base profile of the column
        - desc_stats : statistical or categorical description
        - freq_dist : top-N frequency breakdown

    Raises
    ------
    ValueError
        If the specified column does not exist in the DataFrame.

    Examples
    --------
    >>> summary_column(df, "salary", detailing=True, plots=["histogram"])

    >>> col_stats, desc, top_vals = summary_column(
            df,
            "product_category",
            detailing=True,
            plots=["bar"],
            return_dataframes=True
        )

    >>> summary_column(df, "discount", fast_mode=True)

    Notes
    -----
    - The function detects whether the column is numeric or categorical and adapts its metrics accordingly.
    - Outlier detection (z-score) is only applied to numeric features with sufficient variance.
    - Plots are automatically skipped when `fast_mode=True`.

    See Also
    --------
    summary_dataframe : Summarizes all columns of a DataFrame at once.
    preprocess_column : Cleans and transforms a single column based on rules.
    preprocess_dataframe : End-to-end preprocessing pipeline for the entire DataFrame.

    User Guide
    ----------
    🧠 When Should You Use This?
    - You want to audit or explore one column in depth.
    - You're deciding how to impute, encode, or drop a specific column.
    - You want to visualize category frequency or numeric distribution interactively.
    - You're building an automated column-report pipeline (with return_dataframes=True).

    ⚙️ Recommended Use Cases

    1. **Understand a numeric column with outliers:**
       >>> summary_column(df, "loan_amount", detailing=True, plots=["histogram"])

    2. **Explore a categorical feature for feature engineering:**
       >>> summary_column(df, "device_type", top_n=5, detailing=True, plots=["bar"])

    3. **Check for seasonal missingness (e.g., sensors or logs):**
       >>> summary_column(df, "temperature", time_column="timestamp", plots=["missing_trend"])

    4. **Automation or fast analysis at scale:**
       >>> summary_column(df, "user_age", fast_mode=True)

    💡 Tips:
    - Use with `preprocess_column()` to apply encoding or transformation after diagnosis.
    - If `entropy` is very low, the column may have little signal or be constant.
    - use 'fast_mode'=True` for large datasets to skip slow diagnostics.
    - Use `detailing=False` for a quick overview of the column without deep stats.
    - For many zero-variance columns, use `summary_dataframe()` for batch detection.
    - Always use `return_dataframes=True` if building custom reports or logging stats.
    """


    if fast_mode:
        detailing = False
        plots = []

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    plots = plots or []

    column_data = df[column_name]
    data_type = column_data.dtype
    total_count = len(column_data)
    missing_values = column_data.isnull().sum()
    unique_values = column_data.nunique()
    non_missing_values = total_count - missing_values

    desc_stats = column_data.describe(include="all").to_frame()
    additional_stats = {}
    

    if np.issubdtype(data_type, np.number):
        additional_stats["Variance"] = column_data.var()
        additional_stats["IQR"] = column_data.quantile(0.75) - column_data.quantile(0.25)
        additional_stats["Mode"] = column_data.mode().values[0] if not column_data.mode().empty else np.nan
        additional_stats["Min"] = column_data.min()
        additional_stats["Max"] = column_data.max()

        if detailing:
            if non_missing_values > 1:
                additional_stats["Skewness"] = skew(column_data.dropna())
                additional_stats["Kurtosis"] = kurtosis(column_data.dropna())
            else:
                additional_stats["Skewness"] = np.nan
                additional_stats["Kurtosis"] = np.nan

            mean = column_data.mean()
            std = column_data.std()
            additional_stats["Z-score Outlier Count"] = ((np.abs((column_data - mean) / std) > 3).sum()) if std > 0 else 0

    elif data_type == "object" or data_type.name == "category":
        additional_stats["Mode"] = column_data.mode().values[0] if not column_data.mode().empty else "N/A"
        if detailing and unique_values < 10000:
            value_probs = column_data.value_counts(normalize=True)
            additional_stats["Entropy"] = entropy(value_probs, base=2) if unique_values > 1 else 0

    # ✅ Value Counts (Always computed)
    freq_dist = column_data.value_counts(dropna=False).reset_index().head(top_n)
    freq_dist.columns = ["Value", "Count"]
    freq_dist["Percentage"] = (freq_dist["Count"] / total_count * 100).round(2).astype(str) + " %"

    # Summary Table
    summary_table = pd.DataFrame([
        ["Data Type", data_type],
        ["Total Values", total_count],
        ["Non-Missing Values", non_missing_values],
        ["Missing Values", missing_values],
        ["Missing %", round((missing_values / total_count * 100), 2) if total_count > 0 else 0],
        ["Unique Values", unique_values],
    ] + list(additional_stats.items()), columns=["Metric", "Value"])

    if verbose:
        print("\n" + "=" * 100)
        print(f"Analysis for Column: {column_name}")
        print("=" * 100)

        print("\nSummary Statistics:")
        print(tabulate(summary_table, headers="keys", tablefmt="fancy_grid", showindex=False))

        print("\nDescriptive Statistics:")
        print(tabulate(desc_stats, headers="keys", tablefmt="fancy_grid"))

        if not freq_dist.empty:
            print(f"\nTop {top_n} Value Counts:")
            print(tabulate(freq_dist, headers="keys", tablefmt="fancy_grid"))

    # 🔍 Plots (only if detailing=True)
    if detailing:
        if np.issubdtype(data_type, np.number) and "histogram" in plots:
            plt.figure(figsize=(10, 4))
            column_data.hist(bins=30, edgecolor='black')
            plt.title(f"Histogram of {column_name}")
            plt.xlabel(column_name)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif (data_type == "object" or data_type.name == "category") and "bar" in plots:
            if not freq_dist.empty:
                plt.figure(figsize=(10, 4))
                freq_dist.plot(kind="bar", x="Value", y="Count", legend=False)
                plt.title(f"Top {top_n} Categories in {column_name}")
                plt.xticks(rotation=45, ha='right')
                plt.ylabel("Count")
                plt.tight_layout()
                plt.show()

        if time_column and time_column in df.columns and "missing_trend" in plots:
            if pd.api.types.is_datetime64_any_dtype(df[time_column]):
                missing_series = df.set_index(time_column)[column_name].isnull().resample("W").mean()
                missing_series.plot(figsize=(10, 3), title=f"Missing Rate Over Time for {column_name}")
                plt.ylabel("Missing %")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    if return_dataframes:
        return summary_table, desc_stats, freq_dist
