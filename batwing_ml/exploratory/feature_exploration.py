import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_bool_dtype, CategoricalDtype
from IPython.display import display, HTML
from typing import Optional, List


def display_scrollable_table(df: pd.DataFrame, title: str = "Preview"):
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
        .scroll-table tr:nth-child(even) {{ background-color: #252525; }}
        .scroll-table tr:nth-child(odd) {{ background-color: #1e1e1e; }}
    </style>
    <h3 style='color:#f0f0f0;'>{title}</h3>
    <div class="scroll-table">{html}</div>
    """
    display(HTML(styled_html))


def feature_exploration(
    df: pd.DataFrame,
    target: Optional[str] = None,
    task: Optional[str] = None,
    top_n: int = 20,
    corr_threshold: float = 0.95,
    skew_threshold: float = 1.0,
    sample_size: Optional[int] = None,
    fast_mode: bool = False,
    export_path: Optional[str] = None,
    tree_importance: bool = True,
    perm_importance: bool = True,
    show_preview: bool = True,
    return_summary: bool = False,
    heavy_ops_sample: int = 5000,
    plots: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Quickly analyze the quality and behavior of your dataset's features, with rich statistical summaries,
    automated warnings, optional modeling, and beautiful visualizations.

    This function helps you discover:
    - What features matter most for prediction?
    - Which features are redundant or constant?
    - Which columns may require transformation (e.g., skewed)?
    - How do categorical features relate to the target?

    Ideal for data scientists, analysts, and ML engineers who want to explore datasets before modeling.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset. Each column is treated as a potential feature.

        ✅ Required: Must be a clean rectangular DataFrame (no nested structures).

    target : str, optional
        The name of the target (dependent) variable in your dataset.

        Required for supervised diagnostics like:
        - Mutual Information (MI)
        - Correlation with target
        - Tree-based feature importance
        - Permutation importance
        - Grouped statistics for categorical features

        Example:
        >>> target = "SalePrice"

    task : {'regression', 'classification', 'multiclass'}, optional
        Type of ML problem:
        - 'regression': Predicting continuous values (e.g., prices, temperature)
        - 'classification': Binary prediction (e.g., spam or not spam)
        - 'multiclass': More than 2 classes (e.g., sentiment = low/medium/high)

        ✅ If not passed, the function will auto-detect using target column data type and uniqueness.

    top_n : int, default=20
        How many top features to display in:
        - Summary table
        - Importance plots
        - Skewed distribution visualizations

        Tip: Increase to 50 or 100 for wide datasets with many features.

    corr_threshold : float, default=0.95
        Threshold above which two numeric features are considered highly correlated (redundant).

        - Used to flag potential multicollinearity in the summary
        - Helps you decide which features might be dropped

        Example:
        - A corr of 0.98 between 'age' and 'years_since_birth' may signal redundancy

    skew_threshold : float, default=1.0
        Skewness threshold for numeric features.

        - Columns with absolute skew > threshold are flagged
        - These may benefit from transformation like `log(x+1)` or power scaling

        Tip: Highly skewed features can reduce model performance (especially linear models).

    sample_size : int, optional
        If set, the DataFrame is sampled down to this number of rows before computing summaries.

        Useful for:
        - Large datasets (100k+ rows)
        - Speeding up exploration
        - Limiting memory usage

        Example:
        >>> sample_size = 10000

    fast_mode : bool, default=False
        Turns off all compute-intensive operations:
        - Mutual information
        - Tree-based model fitting
        - Permutation importance
        - Plots

        ✅ Use this for quick diagnostics on very large datasets or inside pipelines.

    export_path : str, optional
        If set, saves the summary table as a CSV.

        Example:
        >>> export_path = "feature_summary.csv"

    tree_importance : bool, default=True
        If enabled and target is provided:
        - Fits a RandomForest model
        - Extracts feature importances
        - Displays top-N most predictive features

        ⚠️ Ignored if `fast_mode=True`.

    perm_importance : bool, default=True
        If enabled:
        - Computes permutation-based feature importance (model-agnostic)
        - More stable than tree importances

        ⚠️ Slower. Use only for small to mid-sized datasets or when needed.

    show_preview : bool, default=True
        Displays the output table in a scrollable, styled HTML block (dark-theme compatible).

        Recommended for:
        - Jupyter Notebook
        - Google Colab
        - VSCode notebooks

    return_summary : bool, default=False
        If True, the summary table (a DataFrame) is returned.

        Useful when you want to:
        - Save to Excel/CSV manually
        - Merge with other reports
        - Visualize in another tool

    heavy_ops_sample : int, default=5000
        For compute-heavy steps like model fitting, mutual info, or permutation:
        - Only this many rows are sampled from the DataFrame

        Keeps everything fast and memory-efficient.

    plots : list of str, optional
        List of visualizations to include. Choose any of:

        - 'importance': Bar plot of top features by tree or permutation importance
        - 'correlation': Heatmap of correlation between numeric features
        - 'skewed': Histogram of skewed numeric columns
        - 'grouped': Bar plots of mean target by category (only for regression)

        Example:
        >>> plots = ["importance", "correlation", "skewed"]

    Returns
    -------
    pd.DataFrame or None
        - If `return_summary=True`: Returns a summary DataFrame with suggestions
        - If `return_summary=False`: Displays summary and plots only

    Examples
    --------
    ▶️ Basic usage:
    >>> feature_exploration(df, target="SalePrice", task="regression")

    ▶️ With visuals and full scoring:
    >>> feature_exploration(
            df,
            target="target",
            task="regression",
            plots=["importance", "correlation", "skewed", "grouped"]
        )

    ▶️ Fast mode scan:
    >>> feature_exploration(df, fast_mode=True)

    ▶️ Export results:
    >>> feature_exploration(df, target="target", export_path="summary.csv")

    ▶️ Capture summary in a variable:
    >>> summary_df = feature_exploration(df, return_summary=True)

    When to Use
    -----------
    ✅ Before modeling: to identify top features, poor features, or potential issues  
    ✅ After cleaning: to detect skew, high cardinality, or multicollinearity  
    ✅ In pipelines: to auto-generate feature insight reports  
    ✅ In dashboards: to track data quality in production  

    Notes
    -----
    - Correlation = Pearson for numeric features
    - Skewness is calculated using SciPy’s `skew()`
    - Importance plots use RandomForest or Permutation models
    - Grouped stats only apply to regression targets and categorical features

    Related
    -------
    • feature_engineering() – to act on features after diagnosing  
    • preprocess_dataframe() – for cleaning before feature exploration  
    • summary_dataframe() – to get statistical overview of the full DataFrame  
    • evaluate_classification_model() – to inspect how features affect model accuracy
    """

    df = df.copy()
    plots = plots or []
    summary = []

    if target:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        X = df.copy()
        y = None

    if not task and target:
        y_nunique = y.nunique()
        if is_numeric_dtype(y) and y_nunique > 10:
            task = "regression"
        elif y_nunique == 2:
            task = "classification"
        else:
            task = "multiclass"

    if fast_mode:
        tree_importance = False
        perm_importance = False
        show_preview = False
        plots = []

    heavy_df = df
    if len(df) > heavy_ops_sample:
        heavy_df = df.sample(n=heavy_ops_sample, random_state=42)

    for col in X.columns:
        ser = X[col]
        dtype = ser.dtype
        n_missing = ser.isnull().sum()
        n_unique = ser.nunique()
        is_num = is_numeric_dtype(ser)
        is_bool = is_bool_dtype(ser)
        is_cat = isinstance(dtype, CategoricalDtype) or is_object_dtype(ser)

        entropy_val, skewness, corr, mi = None, None, None, None
        tree_imp, perm_imp = None, None
        action, reason, comment = "-", "-", "-"

        if is_num and not is_bool and ser.dropna().nunique() > 1:
            try:
                skewness = skew(ser.dropna())
            except:
                skewness = None
            if target and task == "regression" and not fast_mode:
                try:
                    corr = ser.corr(y)
                except:
                    pass

        suggestion = []
        if n_unique == 1:
            suggestion.append("Drop: Constant")
            action, reason, comment = "drop", "constant", "No variance"
        elif n_unique > 100 and is_cat:
            suggestion.append("High cardinality")
            action, reason, comment = "review", "high_cardinality", "Too many unique categories"
        if skewness and abs(skewness) > skew_threshold:
            suggestion.append("Skewed")
            if action == "-":
                action, reason, comment = "transform", "skewed", "Consider log or robust scaling"
        if is_num and n_unique < 10:
            suggestion.append("Discrete numeric")
            comment = "Might be categorical"

        summary.append({
            "Feature": col,
            "Type": "Numeric" if is_num else "Categorical",
            "Missing %": round(n_missing / len(df) * 100, 2),
            "Unique": n_unique,
            "Skewness": round(skewness, 2) if skewness is not None else None,
            "Corr. w/ Target": round(corr, 4) if corr is not None else None,
            "Tree Importance": tree_imp,
            "Perm Importance": perm_imp,
            "Suggestion": ", ".join(suggestion) if suggestion else "-",
            "Action": action,
            "Reason": reason,
            "Comment": comment
        })

    summary_df = pd.DataFrame(summary)

    if target and not fast_mode and (tree_importance or perm_importance):
        model_df = heavy_df.drop(columns=[target])
        model_y = heavy_df[target]
        model_X = model_df.select_dtypes(include=[np.number]).fillna(0)

        if model_X.shape[1] > 0:
            if task == "classification":
                model_y = LabelEncoder().fit_transform(model_y)
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(random_state=42, n_jobs=-1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(model_X, model_y)

            if tree_importance:
                ti_dict = dict(zip(model_X.columns, model.feature_importances_))
                summary_df["Tree Importance"] = summary_df["Feature"].map(ti_dict)

            if perm_importance:
                try:
                    perm = permutation_importance(model, model_X, model_y, n_repeats=5, random_state=42, n_jobs=-1)
                    pi_dict = dict(zip(model_X.columns, perm.importances_mean))
                    summary_df["Perm Importance"] = summary_df["Feature"].map(pi_dict)
                except:
                    summary_df["Perm Importance"] = np.nan

    if export_path:
        summary_df.to_csv(export_path, index=False)

    # 🌐 Plots
    if "importance" in plots:
        imp_cols = ["Tree Importance", "Perm Importance"]
        for col in imp_cols:
            if col in summary_df.columns:
                top_imp = summary_df.dropna(subset=[col]).nlargest(top_n, col)
                plt.figure(figsize=(8, 5))
                sns.barplot(data=top_imp, x=col, y="Feature", color="teal")
                plt.title(f"Top {top_n} Features by {col}")
                plt.tight_layout()
                plt.show()

    if "correlation" in plots:
        corr_data = df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=False, cmap="coolwarm", mask=np.triu(np.ones_like(corr_data, dtype=bool)))
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    if "skewed" in plots:
        skewed_cols = summary_df.loc[summary_df["Skewness"].abs() > skew_threshold, "Feature"].tolist()
        for col in skewed_cols[:top_n]:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution: {col}")
            plt.tight_layout()
            plt.show()

    if "grouped" in plots and task == "regression":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols[:top_n]:
            try:
                gp = df.groupby(col)[target].mean().sort_values()
                plt.figure(figsize=(8, 4))
                sns.barplot(x=gp.index, y=gp.values)
                plt.title(f"Grouped Target Mean: {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            except:
                continue

    if show_preview:
        display_scrollable_table(summary_df.head(top_n), title="🧠 Feature Exploration Summary")

    return summary_df if return_summary else None
