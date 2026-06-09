# DDD Streamlit analysis app — guide

This document explains **how Streamlit runs this project** and how the analysis code is organized.  
Run the app:

```bash
streamlit run streamlit_analysis_app.py
```

The in-app **Guide** tab repeats the weight formulas and a short flow summary.

---

## 1. What Streamlit does

Streamlit is a **script that reruns from top to bottom** whenever you interact with the UI (change a slider, click a button, edit text).

- There is no classic “request handler per route”.
- State survives reruns via **`st.session_state`** (Python dict) and **`@st.cache_data`** (disk/memory cache keyed by function arguments).

Implication: put expensive work (TensorFlow inference, reading thousands of images) **behind buttons** or **cached functions**, not at module import time.

---

## 2. Startup flow (`main()`)

```
streamlit run streamlit_analysis_app.py
        │
        ▼
main()  ──► st.set_page_config(...)
        │
        ├── Sidebar: project root, fold_datasets/, dataset/, image size
        ├── Sidebar: run registry (label=folder lines + auto-discovered runs/)
        │
        └── st.tabs([ Guide, Overview, GradCAM, Training, ... ])
                 │
                 └── each tab calls _tab_*()
```

### Sidebar inputs

| Widget | Purpose |
|--------|---------|
| Project root | Repo root (`runs/`, `weights/`, `fold_datasets/`) |
| Fold JSON dir | `fold_datasets/fold_k.json` manifests |
| Dataset dir | Image files referenced by manifests |
| Image H/W | Resize for models and GradCAM |
| Run registry | `baseline=baseline` → display name + folder under `runs/` |

Registry is merged with folders found under `runs/` (`merge_registry` in `streamlit_ui/helpers.py`).

---

## 3. File roles

| File | Responsibility |
|------|----------------|
| `streamlit_analysis_app.py` | **Entry point** (thin wrapper) |
| `streamlit_ui/analysis_app.py` | All UI: tabs, widgets, buttons, calling core + plots |
| `streamlit_ui/analysis_core.py` | Data: metrics CSV, weight JSON, val batch inference, focus BG table |
| `streamlit_ui/helpers.py` | Matplotlib figures, `st_image`/`st_dataframe` compatibility, registry parsing |
| `streamlit_ui/experiment_core.py` | Experiment launcher (presets, subprocess jobs) |

Package folder is `streamlit_ui/` (not `streamlit/`) so it does not shadow the PyPI `streamlit` library.

| `scripts/auto_optimize_gradcam_weights.py` | Focus ratios + reward weights + α search |
| `scripts/compute_fold_weights.py` | Per-fold `fold_k_weights.json` for all folds |

The app **does not train** unless you use the **Experiments** tab (or run scripts in the terminal).

---

## 4. Tab-by-tab behavior

### Guide
Static markdown + LaTeX for weight math; optional embedded copy of this README.

### Overview
Reads `runs/<run>/fold_k/` summaries and config — quick comparison table.

### GradCAM & mask
- **GradCAM:** pick val image + models → load `.h5` per model, heatmap overlay.
- **Mask:** landmark ROI preview (no model).

### Training
Plots `training_metrics.csv` per run/fold.

### Val reports & CM
Shows saved `val_evaluation_report.txt` and plot PNGs from training.

### Weights
Loads `weights*/fold_k_weights.json`, wide comparison table, **per-class mean/std text** per training run and per weight directory.

### Focus & mask BG
One model: P(drowsy) once; focus ratio at `bg=0` vs `bg=0.2` (mask does not change prediction).

### Decisions
1. Click **Run val inference** → `@st.cache_data` `_cached_val_decisions(...)`.
2. Loads every registry run that has `runs/<run>/fold_k/models/fold_k.h5`.
3. Stores rows in `st.session_state["decision_rows"]`.
4. Compare run vs baseline: flip table, pies, GradCAM on changed samples, confusion matrix.

### Overlap artifacts
Reads precomputed `artifacts/overlap_accuracy_comparison/`.

### Experiments
Starts preset jobs (`compute_weights`, `cv_train`, overlap scripts).

---

## 5. `pd` (pandas) — why it appears in the code

Many tabs build tables from Python lists:

```python
rows = [{"sample": "a.png", "weight": 0.95}, ...]
df = pd.DataFrame(rows)   # columns: sample, weight
```

`DataFrame` supports:

- `df.rename(columns=...)`
- `df["col"].astype(int)`
- `st.dataframe(df)` for interactive tables

In `streamlit_ui/analysis_app.py`:

- `pd` is imported inside `try/except`; if missing, `pd = None`.
- **`require_pandas()`** stops the tab with an install message.
- **`rows_to_dataframe(rows)`** is the preferred constructor.
- Type hints use **`DataFrame`**: real `pd.DataFrame` when type-checking, `Any` at runtime so linters do not require pandas to be installed.

Functions like `_decision_display_df(df, ...)` expect a **DataFrame**, not a list. Callers must convert first:

```python
df = rows_to_dataframe(st.session_state["decision_rows"])
display_df = _decision_display_df(df, registry, baseline_run)
```

---

## 6. Caching and session state

### `@st.cache_data` (example: val inference)

```python
@st.cache_data(show_spinner="...")
def _cached_val_decisions(project_root_str, fold_id, ...):
    ...
    return build_val_decision_rows(...)
```

Cache key = all arguments. Change fold or registry → new inference.  
Clear cache: Streamlit menu **Clear cache** if results look stale.

### `st.session_state`

```python
if st.button("Run val inference"):
    st.session_state["decision_rows"] = rows
rows = st.session_state.get("decision_rows")
```

Keeps the table when you switch tabs or tweak unrelated widgets.

---

## 7. Sample weight pipeline (formulas)

Implemented in `scripts/auto_optimize_gradcam_weights.py`. Summary:

### Step A — Focus ratio per training image

1. Load baseline fold model.
2. GradCAM at **true class** → heatmap \(h\), normalize by max.
3. Landmark mask \(m\) (background = `background_mask_value`).
4. Focus ratio:

\[
r = \frac{\sum_i h_i m_i}{\sum_i h_i + \varepsilon}
\]

### Step B — Reference per fold (current code)

\[
r_{\mathrm{ref}} = P_{60}(\{r_i\})
\]

(60th percentile of train focus ratios in that fold.)

### Step B2 — Legacy `target_focus` (older `weights/`)

Still computed in code for logging, but **today’s** `r_ref` in the weight formula is **P60**, not `target_focus`:

\[
\mathrm{offset} = \mathrm{clip}(0.5 \cdot \sigma_r,\; 0.05,\; 0.15)
\]
\[
r_{\mathrm{target}} = \mathrm{clip}(\mathrm{median}(r) + \mathrm{offset},\; 0.45,\; 0.92)
\]

**Why old weights were often narrow (~0.93–1.08):**

- The **0.45 floor** makes \(r_{\mathrm{ref}}\) artificially high vs typical \(r_i\).
- Most \(\delta_i = r_i - r_{\mathrm{ref}}\) are negative → many \(w_i\) pile up near **clip_min**.
- **bg = 0.2** lowers focus ratios → same effect.

**Why percentile + bg=0 spreads more (~0.6–2.7):**

- P60 tracks the empirical distribution (no 0.45 floor).
- More mixed \(\delta\) signs; α can be larger with meaningful std.
- **bg = 0** increases ROI focus scores → larger positive \(\delta\) for good samples.

Ranking of samples is often **unchanged**; mainly **magnitude** of weights changes.

### Step C — Reward weight

\[
\delta_i = r_i - r_{\mathrm{ref}}, \quad
w_i = \mathrm{clip}(1 + \alpha \delta_i,\, w_{\min},\, w_{\max})
\]

- α: grid search (~0.3–3) maximizing weight std while penalizing clipping and mean drift from 1.
- Default clip uses factor `0.8 * α` around 1.

### Step D — Optional transforms (not used in default CV training)

- **log:** \(\log w_i\) shifted so minimum is 0 → `fold_k_log_weights.json`
- **exp:** \(w_i^{\mathrm{exp}} = \exp(\min(w_i, 50))\) → `fold_k_exp_weights.json`

**Meaning of `exp(min(w_i, 50))`:** apply `exp` to each reward weight; cap the input at 50 only to prevent numeric overflow. This **amplifies** differences (e.g. \(w=1 \Rightarrow e \approx 2.72\), \(w=2 \Rightarrow e^2 \approx 7.39\)). For analysis/alternate experiments — **not** the default CV path.

### Step E — Training

`cv_main.py` reads `weights_dir/fold_k_weights.json` as `sample_weight` on **train** only.  
Baseline run has no `weights_dir` → all weights 1.

---

## 8. Dynamic models / missing fold weights

The decisions table includes a run **only if** this file exists:

`runs/<run_name>/fold_<k>/models/fold_<k>.h5`

If `reward_percentile_dark` fold 5 was never trained, that column is absent — not a Streamlit bug.  
The **Models in this inference** table shows `fold model exists` vs `in inference table`.

---

## 9. Common issues

| Symptom | Cause / fix |
|---------|-------------|
| `NameError: compare_runs` | Old cached script on server — sync latest `streamlit_ui/analysis_app.py` |
| `pd` / pandas errors | `pip install pandas` in the same env as Streamlit |
| Empty compare column | Missing `.h5` for that fold — finish CV training |
| Slow rerun | Normal; use buttons + cache; reduce “Max val samples” |
| `use_container_width` TypeError | Old Streamlit — helpers fall back to `use_column_width` |

---

## 10. Deploying to the cluster

Copy the repo (or sync changed files) to e.g. `/SPACE/spin02/DDD/DDD_cross_validation/`, same conda env (`tf210`), then:

```bash
streamlit run streamlit_analysis_app.py --server.port 8501
```

Set sidebar **Project root** to that path if it differs from the machine where the file was edited.
