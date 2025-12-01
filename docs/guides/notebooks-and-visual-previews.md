# Notebooks & Visual Previews

Batwing-ML is designed to feel natural inside **Jupyter Notebook**, **JupyterLab**, and **Google Colab**. Many of its utilities return rich, styled HTML objects or generate diagnostic plots that display cleanly without extra setup.

This guide walks you through how to get the best experience when using Batwing-ML inside interactive environments.

---

## 1. Rich HTML summaries

Functions like `summary_dataframe`, `summary_column`, and `validate_and_clean_data` generate objects that render as HTML when displayed in a notebook.

### Example

```python
from batwing_ml import summary_dataframe
summary = summary_dataframe(df_clean)
summary  # automatically displays styled HTML
```

These previews are:

* Scrollable
* Syntax-highlighted
* Theme-aware (light/dark)
* Designed to fit in notebook cells without overwhelming the screen

### Tip

If you're not seeing HTML rendering, ensure you're **not** wrapping the output in `print()` — simply place the variable on the last line of a cell.

---

## 2. Column‑level summaries

This is useful when exploring tricky features.

```python
from batwing_ml import summary_column
summary_column(df_clean["age"])
```

You’ll typically see:

* Basic stats
* Distribution preview
* Missing value summary
* Type inference details

These show up as compact, readable HTML blocks.

---

## 3. Previewing preprocessing steps

When you run `preprocess_dataframe`, the optional metadata (`prep_steps`) can also be displayed in notebooks.

```python
X_processed, prep_steps = preprocess_dataframe(
    df_clean.drop(columns=["label"]),
    encode="onehot",
    scale="standard",
    return_steps=True,
)

prep_steps  # view metadata in notebook
```

This helps you verify:

* Which columns were encoded
* Which were scaled
* Which were dropped (and why)
* How missing values were handled

Great for debugging and reproducibility.

---

## 4. Feature engineering previews

Feature engineering often modifies many columns. The metadata object returned (`fe_meta`) displays nicely.

```python
X_fe, fe_meta = feature_engineering(X_processed)
fe_meta
```

You’ll typically see:

* Features added or removed
* Methods applied (variance, MI, model-based)
* Dimensionality-reduction notes

This is extremely useful when experimenting.

---

## 5. Plots in notebooks

Evaluation utilities produce plots automatically:

* ROC / PR curves (classification)
* Confusion matrices
* Residual plots (regression)
* Error distribution plots
* Predicted vs Actual scatter plots

Just call the evaluation function — plots appear inline.

### Example

```python
from batwing_ml import evaluate_classification_model
metrics = evaluate_classification_model(
    model=best_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
```

If you're in **Colab**, you don’t need special setup — matplotlib plots render automatically.

---

## 6. Displaying multiple plots cleanly

Use additional notebook tools to manage layout:

### JupyterLab

```python
%matplotlib inline
```

### Side‑by‑side plots

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
```

### Collapsible sections

In JupyterLab/Notebook:

```python
from IPython.display import HTML
HTML('<details><summary>Show plots</summary> ... </details>')
```

Useful for long evaluation runs.

---

## 7. Using Batwing-ML with pandas‑profiling / ydata‑profiling

Batwing-ML plays nicely with other notebook EDA tools.

```python
from ydata_profiling import ProfileReport

profile = ProfileReport(df_clean, minimal=True)
profile.to_widgets()
```

This complements the lightweight Batwing-ML summaries.

---

## 8. Exporting visuals

Many evaluation functions return a dictionary that includes figures.

You can save them manually:

```python
fig = metrics["residual_plot"]
fig.savefig("residuals.png", dpi=300)
```

Or save entire diagnostic sets per experiment.

---

## 9. Debugging inside notebooks

Interactive environments make debugging easier.

### Check shapes

```python
X.shape, X_processed.shape, X_fe.shape
```

### Inspect encoding

```python
prep_steps["encoded_columns"]
```

### Check for leakage

```python
corr = df_clean.corr(numeric_only=True)[target_col].sort_values()
corr.tail()
```

### Preview heavy operations

If a step feels slow, test it on a smaller slice:

```python
df_clean.head(1000)
```

This can give quick feedback before running the full pipeline.

---

## 10. Tips for a smooth notebook experience

### ✔️ Keep metadata objects visible

They tell you exactly what preprocessing/engineering happened.

### ✔️ Use `fast_mode` early

Notebook runs should be fast.

### ✔️ Use rich display, not print()

Let HTML objects render naturally.

### ✔️ Use Colab GPU/TPU only if needed

Most Batwing-ML tasks are CPU‑friendly.

### ✔️ Re-run cells top‑to‑bottom periodically

Ensures your pipeline is clean and reproducible.

---

## Summary

Interactive notebooks are the best environment for experimenting with Batwing-ML. Use the built‑in HTML previews, plot-generating evaluators, and metadata objects to inspect, debug, and understand every step of your pipeline.

Once the workflow is stable, you can move the code into scripts or a production environment knowing exactly how each step behaves.
