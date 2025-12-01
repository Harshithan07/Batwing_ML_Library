import pandas as pd
import numpy as np
from typing import Optional, Literal, List, Dict, Tuple, Union
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tabulate import tabulate

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


class _FeatureEngineeringSteps(dict):
    def _repr_html_(self):
        rows = []
        for key, value in self.items():
            summary = f"{len(value)} items" if isinstance(value, list) and value else "[✔]"
            details = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append((key, summary, details))

        table_html = tabulate(rows, headers=["Step", "Summary", "Details"], tablefmt="unsafehtml")
        return f"<h3>🔧 Feature Engineering Summary</h3>{table_html}"

    def __str__(self):
        rows = []
        for key, value in self.items():
            summary = f"{len(value)} items" if isinstance(value, list) and value else "[✔]"
            details = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
            rows.append((key, summary, details))
        return tabulate(rows, headers=["Step", "Summary", "Details"], tablefmt="fancy_grid")


def feature_engineering(
    df: pd.DataFrame,
    target: Optional[str] = None,
    mode: Literal["selection", "engineering", "both"] = "both",
    task: Optional[str] = None,
    apply_changes: bool = True,
    strategy: Literal["auto", "manual"] = "auto",
    top_k: Optional[int] = None,
    fast_mode: bool = False,
    show_preview: bool = True,
    return_metadata: bool = True,
    pca_components: Optional[int] = None,
    clustering: bool = False,
    cluster_k: Optional[int] = None,
    cluster_feature_name: str = "cluster_label",
    plots: Optional[List[str]] = None,
    selection_method: Optional[str] = "auto",
    selection_threshold: float = 0.01
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Perform automatic feature selection, engineering, and clustering in one unified function.

    This function is built for data scientists, analysts, or ML engineers who want to streamline the
    feature preprocessing pipeline with just one function call — while maintaining transparency, reproducibility,
    and full control over what happens to their features.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset where each row is an observation and each column is a feature.
        This is the only required argument — all others are optional and context-dependent.

    target : str, optional
        The name of the target column (i.e., label or output variable).
        This is necessary for supervised selection techniques like:
        - Mutual Information (`mutual_info`)
        - Model-based importance (`model`)
        - Lasso regression (`lasso`)
        - Recursive Feature Elimination (`rfe`)

        ⚠️ If you don't pass this, selection methods requiring supervision will be skipped.

    mode : {'selection', 'engineering', 'both'}, default='both'
        Controls what parts of the pipeline are executed:
        - 'selection': Only selects features based on some criteria.
        - 'engineering': Only adds new features (e.g., polynomial or PCA).
        - 'both': Runs selection first, then engineering.

        👉 Tip: Use 'selection' during early cleanup. Use 'engineering' later before modeling.

    task : {'regression', 'classification'}, optional
        Type of ML problem. If not specified, it will be inferred from `target`:
        - Regression: If target is numeric with many unique values.
        - Classification: If target is binary or categorical.

        This controls scoring logic for feature selection (e.g., model choice, MI scoring).

    apply_changes : bool, default=True
        - True → Apply selection/engineering steps directly to the data.
        - False → Simulate the pipeline, only generate metadata and preview output.

        ✅ Use `apply_changes=False` if you're testing or auditing steps first.

    strategy : {'auto', 'manual'}, default='auto'
        Controls engineering behavior:
        - 'auto': Automatically applies PolynomialFeatures and PCA if context is appropriate.
        - 'manual': Skip all engineering unless explicitly requested (e.g., you pass `pca_components`).

    top_k : int, optional
        Number of features to retain in methods like RFE or when displaying top-N importances.

        📌 Useful if you want to aggressively reduce feature dimensionality.

    fast_mode : bool, default=False
        Enables performance-safe mode:
        - Disables model fitting, plotting, clustering, and PCA.
        - Ideal for large datasets or batch scripts.

        ⚠️ No importance plots or PCA will run if this is True.

    show_preview : bool, default=True
        Displays a styled scrollable table showing:
        - Which features are kept
        - Which are engineered
        - Which were dropped

        ✔️ Highly recommended in Jupyter/Colab for visual tracking.

    return_metadata : bool, default=True
        If True, returns a `_FeatureEngineeringSteps` object summarizing what happened.

        Contains:
        - `selected_features`
        - `dropped_features`
        - `created_features`
        - `transforms_applied`

    pca_components : int, optional
        Number of Principal Components to add.
        - Applies PCA on numeric features
        - Useful for dimensionality reduction and multicollinearity handling

        📊 If you add `"pca"` to `plots`, it shows a variance-explained curve.

    clustering : bool, default=False
        Enables KMeans clustering on numeric columns.

        ✅ Adds a new column (`cluster_feature_name`) to represent the assigned cluster.

    cluster_k : int, optional
        Number of clusters (k) to use.
        - If None: shows elbow plot of SSE vs k.
        - If set: assigns clusters directly.

    cluster_feature_name : str, default='cluster_label'
        Name of the column that stores the cluster number for each row.
        You can rename this if you’re using multiple clustering passes.

    plots : list of str, optional
        Visualizations to generate (optional, ignored in fast mode):
        - `"importance"`: Bar chart of top RandomForest importances
        - `"pca"`: Line plot showing cumulative variance from PCA
        - `"clusters"`: 2D PCA plot showing colored clusters

        ➕ Add one or more based on your workflow.

    selection_method : str, optional
        Method used for selecting features:
        - 'variance' → Drops constant or low-variance features
        - 'correlation' → Drops one of each highly correlated pair (based on threshold)
        - 'mutual_info' → Mutual Information score against the target
        - 'model' → Tree-based embedded feature importance (RandomForest)
        - 'lasso' → Uses coefficients from Lasso regression (regression only)
        - 'rfe' → Recursive Feature Elimination (Linear/Logistic)
        - 'auto' → Auto-selects based on task (classification/regression)

    selection_threshold : float, default=0.01
        Threshold for filtering-based methods:
        - If using `'correlation'`: Drop if abs(corr) > threshold
        - If using `'mutual_info'`: Keep if MI score > threshold

        📌 Lower values retain more features, higher values are stricter.

    Returns
    -------
    pd.DataFrame
        The transformed dataset with selected and engineered features.

    _FeatureEngineeringSteps (if return_metadata=True)
        A rich summary of steps applied — for logs, reports, and audits.

    Examples
    --------
    ▶️ Full pipeline:
    >>> X, steps = feature_engineering(
            df=raw_df,
            target='target',
            mode='both',
            strategy='auto',
            selection_method='model',
            pca_components=3,
            clustering=True,
            cluster_k=5,
            plots=['importance', 'pca', 'clusters']
        )

    ▶️ Dry run (no changes, just audit):
    >>> X, log = feature_engineering(df, target='target', apply_changes=False)

    ▶️ PCA for dimensionality reduction:
    >>> X, meta = feature_engineering(df, mode='engineering', pca_components=2)

    ▶️ Lightweight filter-based selection:
    >>> X, meta = feature_engineering(df, target='target', mode='selection', selection_method='mutual_info')

    Notes
    -----
    - Use `return_metadata=True` to track what changed — crucial for reproducibility.
    - Feature selection happens before feature creation.
    - All steps are logged and previewed before being applied.
    - PCA and clustering only work on numeric columns.

    See Also
    --------
    • feature_exploration() – For diagnostics before applying changes  
    • preprocess_dataframe() – For imputation, encoding, and basic cleanup  
    • summary_dataframe() – For an overview of raw features  
    • evaluate_classification_model() – To measure final model quality
    """

    df = df.copy()
    plots = plots or []
    X = df.drop(columns=[target]) if target else df.copy()
    y = df[target] if target else None

    metadata = {
        "selected_features": [],
        "dropped_features": [],
        "created_features": [],
        "transforms_applied": []
    }

    if task is None and target:
        y_nunique = y.nunique()
        task = "regression" if y.dtype.kind in "fc" and y_nunique > 10 else "classification"

    if mode in ["selection", "both"] and not fast_mode:
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        X_num = X[num_cols].fillna(0)

        if selection_method in ["auto", "variance"]:
            vt = VarianceThreshold(threshold=0.0)
            vt.fit(X_num)
            selected = X_num.columns[vt.get_support()].tolist()
            dropped = list(set(X.columns) - set(selected))
            metadata["dropped_features"].extend(dropped)
            metadata["selected_features"].extend(selected)
            metadata["transforms_applied"].append("remove_constant")
            if apply_changes:
                X = X[selected]

        if selection_method in ["auto", "model"] and target is not None:
            model = RandomForestRegressor() if task == "regression" else RandomForestClassifier()
            model.fit(X_num, y)
            sfm = SelectFromModel(model, prefit=True, threshold="median")
            selected = X_num.columns[sfm.get_support()].tolist()
            dropped = list(set(X.columns) - set(selected))
            metadata["dropped_features"].extend(dropped)
            metadata["selected_features"] = selected
            metadata["transforms_applied"].append("model_based_selection")
            if apply_changes:
                X = X[selected]
            if "importance" in plots:
                importances = model.feature_importances_
                top_features = pd.Series(importances, index=X_num.columns).nlargest(top_k or 10)
                plt.figure(figsize=(8, 5))
                sns.barplot(x=top_features.values, y=top_features.index, color="teal")
                plt.title("Top Features by RF Importance")
                plt.tight_layout()
                plt.show()

        if selection_method == "correlation":
            corr_matrix = X_num.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > selection_threshold)]
            metadata["dropped_features"].extend(to_drop)
            metadata["transforms_applied"].append("high_correlation_filter")
            if apply_changes:
                X.drop(columns=to_drop, inplace=True)

        if selection_method == "mutual_info" and target:
            mi_func = mutual_info_regression if task == "regression" else mutual_info_classif
            scores = mi_func(X_num, y)
            selected = X_num.columns[scores > selection_threshold].tolist()
            dropped = list(set(X.columns) - set(selected))
            metadata["dropped_features"].extend(dropped)
            metadata["selected_features"] = selected
            metadata["transforms_applied"].append("mutual_info_filter")
            if apply_changes:
                X = X[selected]

        if selection_method == "rfe" and target:
            base_model = LinearRegression() if task == "regression" else LogisticRegression()
            rfe = RFE(base_model, n_features_to_select=top_k or 10)
            rfe.fit(X_num, y)
            selected = X_num.columns[rfe.support_].tolist()
            dropped = list(set(X.columns) - set(selected))
            metadata["dropped_features"].extend(dropped)
            metadata["selected_features"] = selected
            metadata["transforms_applied"].append("rfe")
            if apply_changes:
                X = X[selected]

        if selection_method == "lasso" and task == "regression":
            lasso = LassoCV(cv=5)
            lasso.fit(X_num, y)
            selected = X_num.columns[lasso.coef_ != 0].tolist()
            dropped = list(set(X.columns) - set(selected))
            metadata["dropped_features"].extend(dropped)
            metadata["selected_features"] = selected
            metadata["transforms_applied"].append("lasso")
            if apply_changes:
                X = X[selected]

    if mode in ["engineering", "both"]:
        created = []
        if strategy == "auto" and not fast_mode:
            try:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_poly = poly.fit_transform(X.select_dtypes(include=np.number))
                poly_cols = poly.get_feature_names_out(X.select_dtypes(include=np.number).columns)
                X_poly = pd.DataFrame(X_poly, columns=poly_cols, index=X.index)
                new_cols = [col for col in poly_cols if col not in X.columns]
                created.extend(new_cols)
                X = pd.concat([X, X_poly[new_cols]], axis=1)
                metadata["transforms_applied"].append("polynomial_features")
            except: pass

            if pca_components:
                pca = PCA(n_components=pca_components)
                X_num = X.select_dtypes(include=np.number).fillna(0)
                pca_result = pca.fit_transform(X_num)
                pca_cols = [f"PCA_{i+1}" for i in range(pca_result.shape[1])]
                df_pca = pd.DataFrame(pca_result, columns=pca_cols, index=X.index)
                X = pd.concat([X, df_pca], axis=1)
                metadata["created_features"].extend(pca_cols)
                metadata["transforms_applied"].append("pca")

                if "pca" in plots:
                    plt.figure(figsize=(6, 4))
                    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
                    plt.title("Cumulative Explained Variance (PCA)")
                    plt.xlabel("# of Components")
                    plt.ylabel("Explained Variance")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.show()

            metadata["created_features"].extend(created)

    if clustering and not fast_mode:
        X_num = X.select_dtypes(include=np.number).fillna(0)
        if cluster_k is None:
            sse = []
            for k in range(2, 11):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(X_num)
                sse.append(km.inertia_)
            plt.figure(figsize=(6, 4))
            plt.plot(range(2, 11), sse, marker='o')
            plt.title("Elbow Method for Optimal k")
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("SSE (Inertia)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            km = KMeans(n_clusters=cluster_k, n_init=10, random_state=42)
            labels = km.fit_predict(X_num)
            X[cluster_feature_name] = labels
            metadata["created_features"].append(cluster_feature_name)
            metadata["transforms_applied"].append(f"kmeans_k={cluster_k}")

            if "clusters" in plots and pca_components:
                pca_2d = PCA(n_components=2).fit_transform(X_num)
                plt.figure(figsize=(6, 4))
                sns.scatterplot(x=pca_2d[:, 0], y=pca_2d[:, 1], hue=labels, palette="tab10")
                plt.title(f"Cluster Visualization (k={cluster_k})")
                plt.tight_layout()
                plt.show()

    if show_preview:
        preview_df = pd.DataFrame({
            "Feature": list(X.columns),
            "Source": ["engineered" if f in metadata["created_features"] else "original" for f in X.columns],
            "Keep?": ["yes" if f in metadata["selected_features"] or mode == "engineering" else "no" for f in X.columns]
        })
        display_scrollable_table(preview_df, title="🧪 Feature Engineering Summary")

    return (X, _FeatureEngineeringSteps(metadata)) if return_metadata else X
