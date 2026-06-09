# Scripts layout

Pipeline code lives here. **Streamlit only launches these** (`streamlit_ui/experiment_core.py`); it does not duplicate the logic.

Do **not** put these under `streamlit_ui/` — they are the same commands you run in a terminal.

## `run/` — CLI entry points (start here)

| Script | Purpose |
|--------|---------|
| `run/cv_train.py` | Cross-validation training → `runs/<name>/` |
| `run/compute_weights.py` | GradCAM sample weights → `weights_*/` |
| `run/overlap_comparison.py` | Val overlap / focus vs accuracy → `artifacts/overlap_accuracy_comparison/<tag>/` |
| `run/model_comparison.py` | Per-fold focus metrics → `artifacts/model_comparison/<tag>/` |

Streamlit **Experiments** tab calls these paths.

```bash
python scripts/run/compute_weights.py --base-run baseline --output-dir weights_percentile
python scripts/run/cv_train.py --run-name my_run --weights-dir weights_percentile
python scripts/run/overlap_comparison.py --output-tag default --model-runs baseline=baseline,reward=reward
python scripts/run/model_comparison.py --output-tag default --focus-metric density_gap
```

## Root `scripts/*.py` — implementations & libraries

| Script | Role |
|--------|------|
| `model_comparison_integral.py` | Model comparison engine (imported by `run/model_comparison.py`) |
| `overlap_accuracy_comparison.py` | Overlap engine (imported by `run/overlap_comparison.py`) |
| `compute_fold_weights.py` | Per-fold weight orchestration |
| `auto_optimize_gradcam_weights.py` | P60 / reward weight optimization |
| `gradcam_density_gap_weights.py` | Density-gap weight formula |
| `log_exp_script.py` | log/exp transforms on reward JSON |

## Utilities (optional, not in Experiments UI)

| Script | Role |
|--------|------|
| `generate_landmark_masks.py` | Batch landmark mask generation |
| `_compare_weight_dirs.py` | Compare weight JSON directories |
| `run_model_comparison_folds.py` | Legacy fold runner (prefer `run/model_comparison.py`) |
