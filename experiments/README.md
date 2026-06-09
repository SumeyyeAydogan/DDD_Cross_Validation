# Experiment presets & job logs

- **`presets.json`** — Saved experiment configurations (editable from Streamlit **Experiments** tab).
- **`logs/`** — stdout/stderr from launched jobs (`<job_id>.log`).
- **`jobs.json`** — Recent job metadata (command, status, return code).

## Streamlit

```bash
streamlit run streamlit_analysis_app.py
```

Open the **Experiments** tab to configure, save presets, and run pipeline steps.

## CLI (same commands as the UI)

```bash
# 1) Weights
python scripts/run/compute_weights.py --base-run baseline --output-dir weights_percentile --background-mask-value 0

# 2) Train
python scripts/run/cv_train.py --run-name reward_percentile_dark --weights-dir weights_percentile

# 3) Overlap analysis
python scripts/run/overlap_comparison.py --output-tag default --model-runs baseline=baseline,reward=reward

# 4) Focus integral comparison
python scripts/run/model_comparison.py --fold-count 5
```

Long CV training can take hours; prefer running `run_cv_train` in a terminal for production runs.
