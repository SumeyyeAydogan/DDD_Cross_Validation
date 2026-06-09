"""
DDD cross-validation — detailed experiment analysis (runs, weights, focus, decisions).

  streamlit run streamlit_analysis_app.py
"""
from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from streamlit_ui.analysis_core import (
    CLASS_NAMES,
    build_val_decision_rows,
    collect_fold_run_summaries,
    compare_accuracy_group,
    discover_weights_dirs,
    focus_bg_comparison_records,
    load_overlap_summary_if_exists,
    load_run_config,
    load_weights_map,
    model_cache_key,
    parse_cv_summary,
    parse_evaluation_report,
    rel_path_from_abs,
    runs_with_fold_model,
    summarize_decision_flips,
    weight_records_for_fold,
    weights_dir_for_run,
)
from streamlit_ui.experiment_core import (
    build_command,
    catalog,
    command_preview,
    load_jobs,
    load_presets,
    presets_path,
    save_presets,
    start_job,
    tail_log,
)
from streamlit_ui.helpers import (
    build_weights_comparison_table,
    discover_run_names,
    fold_run_dir,
    list_folds_for_run,
    load_image_uint8,
    load_metrics_csv,
    mask_to_rgb,
    merge_registry,
    model_h5_path,
    overlay_mask_on_image,
    parse_run_registry,
    plot_overlay_metric_fig,
    plot_pie_counts_fig,
    plot_scatter_fig,
    plot_training_history_fig,
    plot_weight_histogram_fig,
    plot_extra_metrics_fig,
    saved_plot_path,
    short_image_label,
    show_figure_in_streamlit,
    st_dataframe,
    st_image,
    st_rerun,
)
from src.cv_dataloader import labels_from_paths
from src.focus_metrics import FOCUS_METRIC_LABELS, compute_focus_score
from src.gradcam import CustomGradCAM
from src.mask_helpers import create_landmark_mask

if TYPE_CHECKING:
    import pandas as pd

    DataFrame = pd.DataFrame
else:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        pd = None  # type: ignore[assignment,misc]
    DataFrame = Any


def require_pandas():
    """Return pandas module or stop the app with a clear message."""
    if pd is None:
        st.error("This view needs **pandas** (`pip install pandas`).")
        st.stop()
    return pd


def rows_to_dataframe(rows: List[Dict[str, Any]]) -> DataFrame:
    """List of dicts (from core) → pandas table for display and column ops."""
    return require_pandas().DataFrame(rows)


def _tf_init_memory_growth() -> None:
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


def _registry_from_sidebar(sidebar, root: Path) -> List[Tuple[str, str]]:
    sidebar.header("Runs & weights")
    default_map = (
        "baseline=baseline\n"
        "reward=reward\n"
        "percentile_dark=reward_percentile_dark\n"
        "log=log"
    )
    run_text = sidebar.text_area("Run registry (`label=folder`)", value=default_map, height=120)
    discovered = discover_run_names(root / "runs")
    registry = merge_registry(parse_run_registry(run_text), discovered)
    if discovered:
        sidebar.caption(f"Discovered: {', '.join(discovered)}")
    return registry


def _show_fig(fig: plt.Figure, width: int = 700) -> None:
    show_figure_in_streamlit(fig, width)


@st.cache_resource(max_entries=12)
def _load_keras_model(h5_abs: str):
    import tensorflow as tf

    return tf.keras.models.load_model(h5_abs, compile=False)


def _registry_label_for_run(registry: List[Tuple[str, str]], run_name: str) -> str:
    for lab, run in registry:
        if run == run_name:
            return lab
    return run_name


def _render_gradcam_column(
    *,
    label: str,
    run_name: str,
    fold_id: int,
    root: Path,
    image_uint8: np.ndarray,
    img_size: Tuple[int, int],
    true_label: Optional[int],
    display_w: int,
) -> None:
    h5 = model_h5_path(root, run_name, fold_id)
    st.markdown(f"**{label}**")
    st.caption(f"`{run_name}`")
    if not h5.is_file():
        st.error("Model weights missing")
        return
    try:
        import tensorflow as tf

        model = _load_keras_model(str(h5.resolve()))
        cam = CustomGradCAM(model)
        image_norm = image_uint8.astype(np.float32) / 255.0
        prob = float(model.predict(image_norm[None, ...], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0
        conf = prob if pred == 1 else (1.0 - prob)
        heatmap = cam.compute_heatmap(image_norm, class_idx=pred)
        heatmap = tf.image.resize(
            heatmap[..., None],
            img_size,
            method="bilinear",
            antialias=True,
        ).numpy()[..., 0]
        overlay = cam.overlay_heatmap(heatmap, image_norm)
        st_image((np.clip(overlay, 0, 1) * 255).astype(np.uint8), width=display_w)
        gt = f" · GT: {CLASS_NAMES[int(true_label)]}" if true_label is not None else ""
        focus_bg = float(st.session_state.get("gradcam_focus_bg", 0.0))
        mask = create_landmark_mask(
            image_uint8,
            img_size,
            background_mask_value=focus_bg,
            landmark_box_half_size=12,
        )
        metric = str(st.session_state.get("focus_metric", "focus_ratio"))
        if mask is not None:
            hm = np.clip(heatmap, 0.0, None).astype(np.float32)
            score, _ = compute_focus_score(hm, mask, metric=metric)
            focus_txt = f" · {metric}: {score:.3f}"
        else:
            focus_txt = ""
        st.caption(
            f"P(drowsy)={prob:.3f} · Pred: {CLASS_NAMES[pred]} ({conf:.3f}){gt}{focus_txt}"
        )
    except Exception as e:
        st.error(str(e))


def _render_multi_model_gradcam(
    *,
    root: Path,
    registry: List[Tuple[str, str]],
    fold_id: int,
    image_path: str,
    img_size: Tuple[int, int],
    model_runs: List[Tuple[str, str]],
    display_w: int = 240,
) -> None:
    image_uint8 = load_image_uint8(image_path, img_size)
    try:
        true_label = int(labels_from_paths([image_path], CLASS_NAMES)[0])
    except Exception:
        true_label = None
    cols = st.columns(len(model_runs))
    for col, (label, run_name) in zip(cols, model_runs):
        with col:
            _render_gradcam_column(
                label=label,
                run_name=run_name,
                fold_id=fold_id,
                root=root,
                image_uint8=image_uint8,
                img_size=img_size,
                true_label=true_label,
                display_w=display_w,
            )


def _pick_models_ui(
    registry: List[Tuple[str, str]],
    *,
    key_prefix: str,
    max_models: int = 6,
) -> List[Tuple[str, str]]:
    if not registry:
        st.error("No models in registry.")
        return []
    n_avail = len(registry)
    n_compare = int(
        st.selectbox(
            "Number of models",
            options=list(range(1, min(max_models, n_avail) + 1)),
            index=min(1, n_avail - 1),
            key=f"{key_prefix}_n",
        )
    )
    opts = [f"{lab} ({run})" for lab, run in registry]
    picks: List[Tuple[str, str]] = []
    cols = st.columns(min(n_compare, 3))
    for i in range(n_compare):
        with cols[i % len(cols)]:
            choice = st.selectbox(
                f"Model {i + 1}",
                options=opts,
                index=min(i, len(opts) - 1),
                key=f"{key_prefix}_m{i}",
            )
        lab, run = choice.split(" (", 1)
        picks.append((lab, run.rstrip(")")))
    return picks


def _df(rows: List[Dict[str, Any]]) -> Any:
    if pd is None:
        return rows
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _cached_fold_summaries(project_root_str: str, run_name: str) -> List[Dict[str, Any]]:
    return collect_fold_run_summaries(Path(project_root_str), run_name)


@st.cache_data(show_spinner="Val inference (may take a few minutes)…")
def _cached_val_decisions(
    project_root_str: str,
    fold_id: int,
    fold_ds_str: str,
    dataset_dir_str: str,
    baseline_run: str,
    registry_runs_json: str,
    max_samples: int,
    batch_size: int,
    img_h: int,
    img_w: int,
) -> List[Dict[str, Any]]:
    from src.fold_functions import load_fold_manifest

    root = Path(project_root_str)
    registry_runs: List[str] = json.loads(registry_runs_json)
    data = load_fold_manifest(fold_id - 1, fold_ds_str)
    val_paths = list(data["val"]["files"])
    val_labels = list(data["val"]["labels"])

    if max_samples > 0 and len(val_paths) > max_samples:
        rng = np.random.default_rng(42 + fold_id)
        idx = rng.choice(len(val_paths), size=max_samples, replace=False)
        val_paths = [val_paths[i] for i in sorted(idx)]
        val_labels = [val_labels[i] for i in sorted(idx)]

    specs: Dict[str, str] = {}
    for run in registry_runs:
        h5 = model_h5_path(root, run, fold_id)
        if h5.is_file():
            specs[run] = str(h5.resolve())

    wdir = weights_dir_for_run(root, baseline_run)
    weights_map = None
    if wdir is not None:
        wjson = wdir / f"fold_{fold_id}_weights.json"
        if wjson.is_file():
            weights_map = load_weights_map(wjson)

    return build_val_decision_rows(
        val_paths=val_paths,
        val_labels=val_labels,
        model_specs=specs,
        dataset_dir=Path(dataset_dir_str),
        weights_map=weights_map,
        baseline_key=baseline_run,
        img_size=(img_h, img_w),
        batch_size=batch_size,
    )


def _tab_overview(root: Path, registry: List[Tuple[str, str]]) -> None:
    st.subheader("Cross-run overview")
    st.markdown(
        """
**Pipeline (short):** compute weights → CV train (`runs/<folder>/`) → model comparison / overlap → inspect here.

**Run registry (sidebar):** `label=folder` lines define which trained runs appear in Experiments dropdowns.  
You can compare **any number** of models: one baseline + multiselect compare runs.

**Focus metrics (model comparison & overlap):**

| Metric | When to use |
|--------|-------------|
| `focus_ratio` | Classic autoopt / percentile weights (heatmap max-normalized) |
| `density_gap` | Models trained with new density-gap weights (**inside − outside**, no +1) |
| `density_gap_shifted` | Same scale as training weights (`1 + inside − outside`) |
| `inside_density` | Runs trained with inside-only emphasis |

Use a **unique output tag** per experiment so results are not overwritten (`artifacts/model_comparison/<tag>/`, `artifacts/overlap_accuracy_comparison/<tag>/`).
        """
    )
    rows: List[Dict[str, Any]] = []
    for label, run_name in registry:
        run_dir = root / "runs" / run_name
        cfg = load_run_config(run_dir)
        cv_path = run_dir / "cv_summary.txt"
        cv = parse_cv_summary(cv_path.read_text(encoding="utf-8")) if cv_path.is_file() else {}
        folds = list_folds_for_run(root, run_name)
        fold_rows = _cached_fold_summaries(str(root), run_name)
        mean_best_acc = float(np.nanmean([r["best_val_accuracy"] for r in fold_rows if r.get("best_val_accuracy") is not None])) if fold_rows else float("nan")
        rows.append(
            {
                "label": label,
                "run": run_name,
                "folds": len(folds),
                "weights_dir": cfg.get("weights_dir", "—"),
                "cv_val_acc_mean": cv.get("val_accuracy_mean"),
                "cv_val_auc_mean": cv.get("val_auc_mean"),
                "mean_best_val_acc": mean_best_acc,
            }
        )
    st_dataframe(_df(rows))

    mc_root = root / "artifacts" / "model_comparison"
    tag_dirs = sorted([p for p in mc_root.iterdir() if p.is_dir()]) if mc_root.is_dir() else []
    if tag_dirs:
        st.markdown("#### Model comparison tags on disk")
        tag_rows = []
        for tdir in tag_dirs:
            info_path = tdir / "run_info.json"
            info: Dict[str, Any] = {}
            if info_path.is_file():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            tag_rows.append(
                {
                    "tag": tdir.name,
                    "focus_metric": info.get("focus_metric", "—"),
                    "baseline_label": info.get("baseline_label", "—"),
                    "has_metrics": (tdir / "fold_metrics.jsonl").is_file(),
                }
            )
        st_dataframe(_df(tag_rows))
        mc_jsonl = tag_dirs[-1] / "fold_metrics.jsonl"
    else:
        mc_jsonl = mc_root / "fold_metrics.jsonl"
        st.info("No model comparison tags yet. Run **Experiments → Model comparison** with an output tag.")

    if mc_jsonl.is_file() and pd is not None:
        lines = [ln.strip() for ln in mc_jsonl.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        recs: List[Dict[str, Any]] = []
        for ln in lines[-2000:]:
            try:
                recs.append(json.loads(ln))
            except Exception:
                continue
        if recs:
            df_mc = pd.DataFrame(recs)
            if "model_label" in df_mc.columns and "mean_focus" in df_mc.columns and "val_accuracy" in df_mc.columns:
                fm = df_mc["focus_metric"].iloc[0] if "focus_metric" in df_mc.columns and len(df_mc) else "focus_ratio"
                st.markdown(f"#### Focus ↔ Accuracy summary (latest tag, metric=`{fm}`)")
                sum_rows = []
                for lbl, grp in df_mc.groupby("model_label"):
                    g = grp.dropna(subset=["mean_focus", "val_accuracy"])
                    corr = float(g["mean_focus"].corr(g["val_accuracy"])) if len(g) >= 2 else float("nan")
                    sum_rows.append(
                        {
                            "model_label": lbl,
                            "fold_points": int(len(g)),
                            "mean_focus": float(g["mean_focus"].mean()) if len(g) else float("nan"),
                            "mean_val_accuracy": float(g["val_accuracy"].mean()) if len(g) else float("nan"),
                            "corr_focus_vs_val_acc": corr,
                        }
                    )
                st_dataframe(pd.DataFrame(sum_rows).sort_values("model_label"))
    else:
        st.info("Focus↔Accuracy summary will appear after running model_comparison experiments.")

    st.markdown("#### Per-fold best validation metrics")
    for label, run_name in registry:
        fold_rows = _cached_fold_summaries(str(root), run_name)
        if fold_rows:
            with st.expander(f"{label} (`{run_name}`)", expanded=False):
                st_dataframe(_df(fold_rows))


def _tab_model_comparison_artifacts(root: Path) -> None:
    st.subheader("Model comparison artifacts")
    art_root = root / "artifacts" / "model_comparison"
    if not art_root.is_dir():
        st.info(
            "No model comparison artifacts yet. Run **Experiments → Model comparison** first "
            "to create `artifacts/model_comparison/<tag>/fold_metrics.jsonl`."
        )
        return
    tags = sorted([p.name for p in art_root.iterdir() if p.is_dir()])
    if not tags:
        st.info("No tagged outputs found under `artifacts/model_comparison/`.")
        return
    tag = st.selectbox("Artifact tag", tags, key="mc_tag")
    art_dir = art_root / tag
    metrics_path = art_dir / "fold_metrics.jsonl"
    if not metrics_path.is_file():
        st.warning(f"`{tag}` exists but `fold_metrics.jsonl` is missing.")
        return

    run_info_path = art_dir / "run_info.json"
    if run_info_path.is_file():
        try:
            run_info = json.loads(run_info_path.read_text(encoding="utf-8", errors="replace"))
            with st.expander("Run info", expanded=False):
                st.json(run_info)
        except Exception:
            st.caption("`run_info.json` exists but could not be parsed.")

    lines = [ln.strip() for ln in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    rows: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    if not rows:
        st.warning("fold_metrics.jsonl exists but no valid JSON rows were parsed.")
        return

    if pd is None:
        st.json(rows[:20])
        return

    df = pd.DataFrame(rows)
    exp_ids = sorted(df["experiment_id"].dropna().astype(str).unique().tolist()) if "experiment_id" in df.columns else []
    if exp_ids:
        exp_pick = st.selectbox("Experiment id", ["(all)"] + exp_ids, key="mc_exp")
        if exp_pick != "(all)":
            df = df[df["experiment_id"].astype(str) == exp_pick]

    labels = sorted(df["model_label"].dropna().astype(str).unique().tolist()) if "model_label" in df.columns else []
    if labels:
        pick_labels = st.multiselect("Model labels", labels, default=labels, key="mc_labels")
        if pick_labels:
            df = df[df["model_label"].astype(str).isin(pick_labels)]

    st.markdown("#### Fold metrics (table)")
    show_cols = [
        c
        for c in [
            "fold_id",
            "model_label",
            "weight_type",
            "focus_metric",
            "mean_focus",
            "val_accuracy",
            "val_auc",
            "delta_P_vs_baseline",
            "delta_Area_hist_vs_baseline",
            "experiment_id",
        ]
        if c in df.columns
    ]
    st_dataframe(df[show_cols].head(1000))

    if {"mean_focus", "val_accuracy", "model_label"}.issubset(df.columns):
        plot_df = df.dropna(subset=["mean_focus", "val_accuracy"]).copy()
        if len(plot_df):
            st.markdown("#### Focus ratio vs validation accuracy")
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for lbl, grp in plot_df.groupby("model_label"):
                ax.scatter(grp["mean_focus"].values, grp["val_accuracy"].values, alpha=0.75, s=36, label=str(lbl))
            ax.set_xlabel("Mean focus ratio")
            ax.set_ylabel("Validation accuracy")
            ax.set_title("Fold points: focus vs val_accuracy")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            _show_fig(fig, 650)

            corr_rows = []
            for lbl, grp in plot_df.groupby("model_label"):
                corr = float(grp["mean_focus"].corr(grp["val_accuracy"])) if len(grp) >= 2 else float("nan")
                corr_rows.append({"model_label": lbl, "n_points": int(len(grp)), "corr_focus_vs_val_acc": corr})
            st_dataframe(pd.DataFrame(corr_rows).sort_values("model_label"))

    st.markdown("#### Saved plots")
    plot_dir = art_dir / "plots"
    if plot_dir.is_dir():
        for p in sorted(plot_dir.glob("*.png")):
            st_image(str(p), caption=p.name, use_container_width=True)
    else:
        st.caption("No `plots/` directory yet.")


def _tab_training(root: Path, registry: List[Tuple[str, str]]) -> None:
    st.subheader("Training history comparison")
    fold_id = int(st.number_input("Fold", 1, 20, 1, key="tr_fold"))
    metric = st.selectbox(
        "Metric to overlay",
        ["val_accuracy", "val_auc", "val_loss", "accuracy", "loss", "val_precision", "val_recall", "auc"],
        index=0,
        key="tr_metric",
    )
    picks = st.multiselect(
        "Runs",
        options=[f"{lab} ({run})" for lab, run in registry],
        default=[f"{lab} ({run})" for lab, run in registry[:3]],
        key="tr_runs",
    )
    if st.button("Plot overlay", type="primary", key="tr_plot"):
        series: Dict[str, List[float]] = {}
        for opt in picks:
            lab, run = opt.split(" (", 1)
            run = run.rstrip(")")
            hist = load_metrics_csv(fold_run_dir(root, run, fold_id) / "training_metrics.csv")
            if hist:
                series[f"{lab}"] = hist
        fig = plot_overlay_metric_fig(series, metric, title=f"{metric} — fold {fold_id}")
        if fig is None:
            st.warning("No data for selected metric/runs.")
        else:
            _show_fig(fig, 900)

    st.markdown("---")
    st.caption("Side-by-side panels (full history + extra metrics)")
    col_run = st.selectbox("Single-run detail", [f"{l} ({r})" for l, r in registry], key="tr_detail_run")
    lab, run = col_run.split(" (", 1)
    run = run.rstrip(")")
    hist = load_metrics_csv(fold_run_dir(root, run, fold_id) / "training_metrics.csv")
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_training_history_fig(hist, title=f"{lab} fold {fold_id}")
        if fig:
            _show_fig(fig, 420)
    with c2:
        fig2 = plot_extra_metrics_fig(hist, title=f"{lab} fold {fold_id}")
        if fig2:
            _show_fig(fig2, 420)


def _tab_val_reports(root: Path, registry: List[Tuple[str, str]]) -> None:
    st.subheader("Validation reports & confusion matrices")
    fold_id = int(st.number_input("Fold", 1, 20, 1, key="vr_fold"))
    picks = st.multiselect(
        "Runs",
        [f"{l} ({r})" for l, r in registry],
        default=[f"{l} ({r})" for l, r in registry[: min(3, len(registry))]],
        key="vr_runs",
    )
    cols = st.columns(len(picks) if picks else 1)
    for col, opt in zip(cols, picks):
        lab, run = opt.split(" (", 1)
        run = run.rstrip(")")
        fold_dir = fold_run_dir(root, run, fold_id)
        with col:
            st.markdown(f"**{lab}**")
            report_path = fold_dir / "plots" / "val_evaluation_report.txt"
            if report_path.is_file():
                text = report_path.read_text(encoding="utf-8")
                parsed = parse_evaluation_report(text)
                st.text_area(
                    "Report",
                    text,
                    height=220,
                    key=f"vr_txt_{run}_{fold_id}",
                )
                if parsed.get("test_accuracy") is not None:
                    st.metric("Test accuracy", f"{parsed['test_accuracy']:.4f}")
                if parsed.get("roc_auc") is not None:
                    st.metric("ROC-AUC", f"{parsed['roc_auc']:.4f}")
            else:
                st.warning("No val_evaluation_report.txt")

            for png_name, caption in (
                ("val_confusion_matrix.png", "Confusion matrix"),
                ("confusion_matrix.png", "Confusion matrix"),
                ("val_roc_curve.png", "ROC"),
                ("val_precision_recall_curve.png", "PR curve"),
                ("training_history.png", "Training history"),
            ):
                png = fold_dir / "plots" / png_name
                if png.is_file():
                    st_image(str(png), caption=caption, use_container_width=True)
                    break


def _tab_weights(
    root: Path,
    fold_ds: Path,
    dataset_dir: Path,
    registry: List[Tuple[str, str]],
    img_size: Tuple[int, int],
) -> None:
    st.subheader("Sample weights explorer")
    wdirs = discover_weights_dirs(root)
    if not wdirs:
        st.warning("No `weights*` directories with fold_*_weights.json found.")
        return

    fold_id = int(st.number_input("Fold", 1, 20, 1, key="wt_fold"))
    split = st.radio("Split", ["train", "val"], horizontal=True, key="wt_split")
    selected_dirs = st.multiselect(
        "Weight sets (one column per set)",
        options=[str(p) for p in wdirs],
        default=[str(wdirs[0]), str(wdirs[1])] if len(wdirs) > 1 else [str(wdirs[0])],
        key="wt_dirs",
    )

    per_set: Dict[str, List[Dict[str, Any]]] = {}
    long_rows: List[Dict[str, Any]] = []
    for wd in selected_dirs:
        wpath = Path(wd)
        recs = weight_records_for_fold(
            weights_dir=wpath,
            fold_id=fold_id,
            fold_datasets_dir=fold_ds,
            dataset_dir=dataset_dir,
            split=split,
        )
        tag = wpath.name
        per_set[tag] = recs
        for r in recs:
            long_rows.append({**r, "weights_set": tag})

    if not long_rows:
        st.warning("No weight records for this fold/split.")
        return

    wide_rows = build_weights_comparison_table(per_set)
    if pd is None:
        st.json(wide_rows[:10])
        return

    wide_df = pd.DataFrame(wide_rows)
    st.markdown("#### Comparison table")
    st_dataframe(wide_df.drop(columns=["rel_path", "abs_path"], errors="ignore").head(800))

    if len(selected_dirs) >= 2:
        tags = [Path(d).name for d in selected_dirs[:2]]
        c1, c2 = f"weight ({tags[0]})", f"weight ({tags[1]})"
        if c1 in wide_df.columns and c2 in wide_df.columns:
            st.caption(
                f"{tags[0]} vs {tags[1]} — corr={wide_df[c1].corr(wide_df[c2]):.4f}, "
                f"Δ mean={(wide_df[c2] - wide_df[c1]).mean():.4f}"
            )
            _show_fig(
                plot_scatter_fig(
                    wide_df[c1].dropna().tolist(),
                    wide_df[c2].dropna().tolist(),
                    title=f"{tags[0]} vs {tags[1]}",
                    xlabel=tags[0],
                    ylabel=tags[1],
                ),
                450,
            )

    long_df = pd.DataFrame(long_rows)
    st.markdown("#### By class per training run (mean ± std)")
    for lab, run_name in registry:
        wd = weights_dir_for_run(root, run_name)
        st.markdown(f"**{lab}** (`{run_name}`)")
        if wd is None:
            st.caption("No `weights_dir` in run config — uniform weight 1.0 during training.")
            for cname in CLASS_NAMES:
                st.write(f"- {cname}: mean **1.0000**, std **0.0000**")
            continue
        recs = weight_records_for_fold(
            weights_dir=wd,
            fold_id=fold_id,
            fold_datasets_dir=fold_ds,
            dataset_dir=dataset_dir,
            split=split,
        )
        if not recs:
            st.caption(f"No weights file under `{wd.name}` for fold {fold_id}.")
            continue
        st.caption(f"Weights: `{wd}`")
        for cname in CLASS_NAMES:
            vals = [r["weight"] for r in recs if r["class_name"] == cname]
            if vals:
                st.write(
                    f"- {cname}: mean **{np.mean(vals):.4f}**, std **{np.std(vals):.4f}**, n={len(vals)}"
                )

    st.markdown("#### By weight directory (mean ± std)")
    for tag in long_df["weights_set"].unique():
        sub = long_df[long_df["weights_set"] == tag]
        st.markdown(f"**{tag}**")
        for cname in CLASS_NAMES:
            vals = sub[sub["class_name"] == cname]["weight"].tolist()
            if vals:
                st.write(
                    f"- {cname}: mean **{np.mean(vals):.4f}**, std **{np.std(vals):.4f}**, n={len(vals)}"
                )
    h1, h2 = st.columns(2)
    for i, tag in enumerate(long_df["weights_set"].unique()):
        sub = long_df[long_df["weights_set"] == tag]
        with (h1 if i % 2 == 0 else h2):
            _show_fig(plot_weight_histogram_fig(sub["weight"].tolist(), title=tag), 380)

    st.markdown("---")
    st.markdown("#### Inspect one sample")
    if "sample" in wide_df.columns:
        pick = st.selectbox("Sample", wide_df["sample"].tolist(), key="wt_pick")
        row = wide_df[wide_df["sample"] == pick].iloc[0]
        with st.expander("Full disk path"):
            st.code(str(row.get("abs_path", "")))
        abs_path = str(row.get("abs_path", ""))
        if abs_path and os.path.isfile(abs_path):
            grad_runs = _pick_models_ui(registry, key_prefix="wt_gc", max_models=4)
            if grad_runs and st.button("GradCAM for selected sample", key="wt_gc_go"):
                _render_multi_model_gradcam(
                    root=root,
                    registry=registry,
                    fold_id=fold_id,
                    image_path=abs_path,
                    img_size=img_size,
                    model_runs=grad_runs,
                )


def _tab_gradcam_and_mask(
    root: Path,
    fold_ds: Path,
    dataset_dir: Path,
    img_size: Tuple[int, int],
    registry: List[Tuple[str, str]],
    default_fold_id: int,
) -> None:
    from src.fold_functions import load_fold_manifest

    sub_gc, sub_mask = st.tabs(["GradCAM (same image)", "ROI mask preview"])

    with sub_gc:
        st.subheader("GradCAM — same image, multiple models")
        fold_id = int(st.number_input("Fold", 1, 20, default_fold_id, key="gc_fold"))
        display_w = int(st.slider("Panel width (px)", 120, 500, 240, key="gc_w"))
        data = load_fold_manifest(fold_id - 1, str(fold_ds))
        val_paths = [str(Path(p)) for p in data["val"]["files"]]
        if not val_paths:
            st.warning("No validation images in fold manifest.")
            return
        labels_display = [
            short_image_label(os.path.relpath(p, str(dataset_dir)).replace("\\", "/"))
            for p in val_paths
        ]
        idx = st.selectbox(
            "Validation image",
            range(len(val_paths)),
            format_func=lambda i: labels_display[i],
            key="gc_img",
        )
        picks = _pick_models_ui(registry, key_prefix="gc", max_models=6)
        if picks and st.button("Show GradCAM", type="primary", key="gc_show"):
            _render_multi_model_gradcam(
                root=root,
                registry=registry,
                fold_id=fold_id,
                image_path=val_paths[idx],
                img_size=img_size,
                model_runs=picks,
                display_w=display_w,
            )

    with sub_mask:
        st.subheader("Landmark ROI mask on image")
        fold_id = int(st.number_input("Fold (val list)", 1, 20, default_fold_id, key="mk_fold"))
        mask_w = int(st.slider("Display width (px)", 120, 400, 220, key="mk_w"))
        data = load_fold_manifest(fold_id - 1, str(fold_ds))
        val_paths = list(data["val"]["files"])
        if not val_paths:
            return
        labels_display = [
            short_image_label(os.path.relpath(str(Path(p)), str(dataset_dir)).replace("\\", "/"))
            for p in val_paths
        ]
        idx = st.selectbox(
            "Image",
            range(len(val_paths)),
            format_func=lambda i: labels_display[i],
            key="mk_img",
        )
        image_uint8 = load_image_uint8(str(val_paths[idx]), img_size)
        roi_half = int(st.slider("Landmark box half-size", 4, 40, 12, key="mk_roi"))
        bg_val = float(st.slider("Background mask value", 0.0, 1.0, 0.0, 0.05, key="mk_bg"))
        mask = create_landmark_mask(
            image_uint8,
            img_size,
            background_mask_value=bg_val,
            landmark_box_half_size=roi_half,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            st.caption("Original")
            st_image(image_uint8, width=mask_w)
        with m2:
            st.caption("Mask")
            if mask is not None:
                st_image(mask_to_rgb(mask), width=mask_w)
            else:
                st.warning("No face detected")
        with m3:
            st.caption("Overlay")
            if mask is not None:
                st_image(overlay_mask_on_image(image_uint8, mask), width=mask_w)
            else:
                st_image(image_uint8, width=mask_w)


def _tab_focus_mask(
    root: Path,
    fold_ds: Path,
    dataset_dir: Path,
    img_size: Tuple[int, int],
    registry: List[Tuple[str, str]],
) -> None:
    st.subheader("Focus ratio vs mask background (bg=0 vs bg=0.2)")
    st.caption(
        "Uses **one** trained model (your choice). The mask background only changes how "
        "**focus ratio** is measured on the same GradCAM heatmap; it does **not** change the model prediction."
    )
    from src.fold_functions import load_fold_manifest

    fold_id = int(st.number_input("Fold", 1, 20, 1, key="fc_fold"))
    run_opt = st.selectbox("Trained model (for GradCAM + P(drowsy))", [f"{l} ({r})" for l, r in registry], key="fc_run")
    lab, run = run_opt.split(" (", 1)
    run = run.rstrip(")")
    h5 = model_h5_path(root, run, fold_id)
    if not h5.is_file():
        st.error(f"Model not found: {h5}")
        return

    data = load_fold_manifest(fold_id - 1, str(fold_ds))
    paths = list(data["val"]["files"])
    n_sample = int(st.slider("Sample size (val)", 10, min(500, len(paths)), 80, key="fc_n"))
    rng = np.random.default_rng(42)
    if len(paths) > n_sample:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), n_sample, replace=False))]

    if st.button("Compute focus ratios", type="primary", key="fc_go"):
        with st.spinner("Computing…"):
            recs = focus_bg_comparison_records(
                model_path=str(h5),
                image_paths=paths,
                img_size=img_size,
                dataset_dir=str(dataset_dir),
                focus_metric=str(st.session_state.get("focus_metric", "focus_ratio")),
            )
        st.session_state["focus_bg_records"] = recs

    recs = st.session_state.get("focus_bg_records")
    if not recs or pd is None:
        if recs:
            st.json(recs[:5])
        else:
            st.info("Click **Compute focus ratios** to fill the table.")
        return

    raw = pd.DataFrame(recs)
    rename = {
        "sample": "Sample",
        "true_class": "True label",
        "pred_class": "Prediction",
        "prob_drowsy": "P(drowsy)",
        "focus_bg_0": "Focus (bg=0)",
        "focus_bg_0.2": "Focus (bg=0.2)",
        "focus_delta_0.2_minus_0": "Δ focus (0.2−0)",
    }
    display = raw.rename(columns=rename)

    valid = display.dropna(subset=["Focus (bg=0)", "Focus (bg=0.2)"])
    st.write(f"Face detected: **{len(valid)}** / {len(display)}")
    if len(valid):
        st.caption(
            f"Δ focus mean={valid['Δ focus (0.2−0)'].mean():.4f} · "
            f"std={valid['Δ focus (0.2−0)'].std():.4f}"
        )
        _show_fig(
            plot_scatter_fig(
                valid["Focus (bg=0)"].tolist(),
                valid["Focus (bg=0.2)"].tolist(),
                title="Focus: bg=0 vs bg=0.2",
                xlabel="bg=0",
                ylabel="bg=0.2",
            ),
            500,
        )

    show_cols = [c for c in rename.values() if c in display.columns]
    st_dataframe(display[show_cols].head(300))

    if "Sample" in display.columns:
        pick = st.selectbox("Inspect sample (GradCAM)", display["Sample"].tolist(), key="fc_pick")
        row_raw = raw[raw["sample"] == pick].iloc[0] if "sample" in raw.columns else raw.iloc[0]
        img_path = str(row_raw.get("image_path", ""))
        with st.expander("Full path"):
            st.code(img_path)
        if img_path and os.path.isfile(img_path):
            st.caption(
                f"P(drowsy)={float(row_raw.get('prob_drowsy', 0)):.3f} · "
                f"Pred={row_raw.get('pred_class', '')} · "
                f"Focus bg=0={row_raw.get('focus_bg_0')} · bg=0.2={row_raw.get('focus_bg_0.2')}"
            )
            if st.button("Show GradCAM", key="fc_gc_btn"):
                _render_multi_model_gradcam(
                    root=root,
                    registry=registry,
                    fold_id=fold_id,
                    image_path=img_path,
                    img_size=img_size,
                    model_runs=[(lab, run)],
                )


def _decision_display_df(df: DataFrame, registry: List[Tuple[str, str]], baseline_run: str) -> DataFrame:
    """Human-readable column names for the decisions table."""
    out = df.copy()
    if "rel_path" in out.columns:
        out["Sample"] = out["rel_path"].map(short_image_label)
    base_lab = _registry_label_for_run(registry, baseline_run)
    rename = {
        "true_class": "True label",
        f"{baseline_run}_pred_class": f"Pred ({base_lab})",
        f"{baseline_run}_prob": f"P(drowsy) ({base_lab})",
        f"{baseline_run}_confidence": f"Confidence ({base_lab})",
        "weight": "Train weight",
    }
    for run in [c.replace("_pred_class", "") for c in out.columns if c.endswith("_pred_class") and c != f"{baseline_run}_pred_class"]:
        lab = _registry_label_for_run(registry, run)
        rename[f"{run}_pred_class"] = f"Pred ({lab})"
        rename[f"{run}_prob"] = f"P(drowsy) ({lab})"
        rename[f"{run}_confidence"] = f"Confidence ({lab})"
        rename[f"flip_{run}"] = f"Decision changed? ({lab})"
        rename[f"acc_delta_{run}"] = f"Accuracy vs {base_lab} ({lab})"
    out = out.rename(columns=rename)
    hide = [
        c
        for c in out.columns
        if c in ("image_path", "rel_path", "abs_path")
        or c.endswith("_pred")
        or c.endswith("_correct")
    ]
    show_first = ["Sample", "True label", "Train weight"]
    rest = [c for c in out.columns if c not in hide and c not in show_first]
    cols = [c for c in show_first if c in out.columns] + sorted(rest)
    return out[cols]


def _add_flip_transition_columns(
    df: DataFrame,
    baseline_run: str,
    cmp_run: str,
) -> DataFrame:
    out = df.copy()
    bp = out[f"{baseline_run}_pred"].astype(int)
    cp = out[f"{cmp_run}_pred"].astype(int)
    tl = out["true_label"].astype(int)
    out["decision_changed"] = (bp != cp).astype(int)
    b_ok = bp == tl
    c_ok = cp == tl
    out["acc_vs_baseline"] = np.select(
        [c_ok & ~b_ok, ~c_ok & b_ok, c_ok & b_ok],
        ["improved", "worsened", "same"],
        default="same",
    )
    flip = out["decision_changed"] == 1
    out["flip_wrong_to_correct"] = (flip & ~b_ok & c_ok).astype(int)
    out["flip_correct_to_wrong"] = (flip & b_ok & ~c_ok).astype(int)
    return out


def _tab_decisions(
    root: Path,
    fold_ds: Path,
    dataset_dir: Path,
    registry: List[Tuple[str, str]],
    img_size: Tuple[int, int],
) -> None:
    st.subheader("Validation decisions & GradCAM")
    st.caption(
        "**Decision changed** = predicted class differs from baseline. "
        "**Accuracy vs baseline** = improved / same / worsened relative to ground truth."
    )

    fold_id = int(st.number_input("Fold", 1, 20, 1, key="dc_fold"))
    baseline_opt = st.selectbox("Baseline run", [f"{l} ({r})" for l, r in registry], key="dc_base")
    baseline_run = baseline_opt.split(" (", 1)[1].rstrip(")")
    compare_opts = st.multiselect(
        "Compare runs",
        [f"{l} ({r})" for l, r in registry if r != baseline_run],
        default=[f"{l} ({r})" for l, r in registry if r != baseline_run][:2],
        key="dc_cmp",
    )
    compare_runs = [o.split(" (", 1)[1].rstrip(")") for o in compare_opts]
    max_samples = int(st.slider("Max val samples (0 = all)", 0, 8358, 1500, 100, key="dc_max"))
    batch_size = int(st.selectbox("Batch size", [16, 32, 64], index=1, key="dc_bs"))

    if st.button("Run val inference", type="primary", key="dc_go"):
        all_runs = [r for _, r in registry]
        rows = _cached_val_decisions(
            str(root),
            fold_id,
            str(fold_ds),
            str(dataset_dir),
            baseline_run,
            json.dumps(all_runs),
            max_samples,
            batch_size,
            img_size[0],
            img_size[1],
        )
        st.session_state["decision_rows"] = rows
        st.session_state["decision_baseline"] = baseline_run
        st.session_state["decision_compare"] = compare_runs
        st.session_state["decision_fold"] = fold_id

    rows = st.session_state.get("decision_rows")
    if not rows:
        st.info("Run inference to populate the table.")
        return

    if pd is None:
        st.json(rows[:20])
        return

    df = rows_to_dataframe(rows)
    fold_id = int(st.session_state.get("decision_fold", fold_id))

    st.markdown("#### Models in this inference (fold `%d`)" % fold_id)
    model_status = runs_with_fold_model(root, registry, fold_id)
    pred_runs = {c.replace("_pred", "") for c in df.columns if c.endswith("_pred")}
    status_rows = []
    for lab, run, ok, path in model_status:
        status_rows.append(
            {
                "label": lab,
                "run": run,
                "fold model exists": ok,
                "in inference table": run in pred_runs,
                "path": path if not ok else "✓",
            }
        )
    st_dataframe(pd.DataFrame(status_rows))
    missing = [r for r in pred_runs if not model_h5_path(root, r, fold_id).is_file()]
    if missing:
        st.warning(
            "No `fold_%d.h5` for: %s — columns omitted from table. Train that fold or pick another run."
            % (fold_id, ", ".join(missing))
        )

    display_df = _decision_display_df(df, registry, baseline_run)
    st.markdown("#### All samples (cached)")
    st_dataframe(display_df.head(400))

    cmp_for_detail = st.selectbox(
        "Detailed breakdown for compare run",
        st.session_state.get("decision_compare", compare_runs),
        format_func=lambda r: _registry_label_for_run(registry, r),
        key="dc_detail_cmp",
    )
    if f"{cmp_for_detail}_pred" not in df.columns:
        st.error(f"No predictions for `{cmp_for_detail}` on fold {fold_id} (model file missing?).")
        return

    df = _add_flip_transition_columns(df, baseline_run, cmp_for_detail)
    cmp_lab = _registry_label_for_run(registry, cmp_for_detail)
    summ = summarize_decision_flips(rows, cmp_for_detail)

    st.markdown(f"#### vs baseline: **{cmp_lab}**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Decision changed", summ["n_flip"])
    m2.metric("Change rate", f"{100 * summ['flip_rate']:.2f}%")
    m3.metric("Accuracy improved", summ["acc_improved"])
    m4.metric("Accuracy worsened", summ["acc_worsened"])
    m5.metric("Median conf. (flips)", f"{summ['flip_confidence_median']:.3f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_pie = plot_pie_counts_fig(
            {"unchanged": summ["n_total"] - summ["n_flip"], "changed": summ["n_flip"]},
            title="Prediction vs baseline",
        )
        if fig_pie:
            _show_fig(fig_pie, 280)
    with c2:
        acc_counts = df["acc_vs_baseline"].value_counts().to_dict()
        fig_acc = plot_pie_counts_fig(
            {
                "Improved": acc_counts.get("improved", 0),
                "Same": acc_counts.get("same", 0),
                "Worsened": acc_counts.get("worsened", 0),
            },
            title="Accuracy vs baseline (GT)",
        )
        if fig_acc:
            _show_fig(fig_acc, 280)
    with c3:
        fig_flip = plot_pie_counts_fig(
            {
                "Wrong→Correct": int(df["flip_wrong_to_correct"].sum()),
                "Correct→Wrong": int(df["flip_correct_to_wrong"].sum()),
                "Other flips": int(
                    df["decision_changed"].sum()
                    - df["flip_wrong_to_correct"].sum()
                    - df["flip_correct_to_wrong"].sum()
                ),
            },
            title="Among changed decisions",
        )
        if fig_flip:
            _show_fig(fig_flip, 280)

    flip_df = df[df["decision_changed"] == 1].copy()
    if len(flip_df):
        st.markdown("#### Changed decisions (table)")
        flip_show = flip_df.copy()
        if "rel_path" in flip_show.columns:
            flip_show["Sample"] = flip_show["rel_path"].map(short_image_label)
        show_flip_cols = [
            c
            for c in [
                "Sample",
                "true_class",
                f"{baseline_run}_pred_class",
                f"{cmp_for_detail}_pred_class",
                f"{baseline_run}_confidence",
                f"{cmp_for_detail}_confidence",
                "acc_vs_baseline",
                "flip_wrong_to_correct",
                "flip_correct_to_wrong",
                "weight",
            ]
            if c in flip_show.columns
        ]
        st_dataframe(flip_show[show_flip_cols].head(200))

    st.markdown("---")
    st.markdown("#### Inspect **changed** sample — GradCAM")
    if len(flip_df) and "rel_path" in flip_df.columns:
        flip_labels = flip_df["rel_path"].map(short_image_label).tolist()
        flip_indices = flip_df.index.tolist()
        pick_local = st.selectbox(
            "Changed sample",
            range(len(flip_indices)),
            format_func=lambda i: flip_labels[i],
            key="dc_flip_sample",
        )
        row = flip_df.loc[flip_indices[pick_local]]
    elif "rel_path" in df.columns:
        st.info("No decision changes in this compare run; showing any sample.")
        labels = df["rel_path"].map(short_image_label).tolist()
        pick_local = st.selectbox("Sample", range(len(labels)), format_func=lambda i: labels[i], key="dc_any_sample")
        row = df.iloc[pick_local]
    else:
        row = None

    if row is not None:
        img_path = str(row.get("image_path", ""))
        with st.expander("Full path"):
            st.code(img_path)
        model_runs_for_gc: List[Tuple[str, str]] = []
        for r in [baseline_run, cmp_for_detail]:
            if f"{r}_pred" in df.columns and model_h5_path(root, r, fold_id).is_file():
                model_runs_for_gc.append((_registry_label_for_run(registry, r), r))
        if img_path and os.path.isfile(img_path) and model_runs_for_gc:
            st.caption(
                " | ".join(
                    [f"**GT:** {row.get('true_class', '')}"]
                    + [
                        f"**{_registry_label_for_run(registry, r)}:** {row.get(f'{r}_pred_class', '')} "
                        f"conf={float(row.get(f'{r}_confidence', 0)):.3f}"
                        for r in [baseline_run, cmp_for_detail]
                        if f"{r}_pred_class" in row.index
                    ]
                )
            )
            _render_multi_model_gradcam(
                root=root,
                registry=registry,
                fold_id=fold_id,
                image_path=img_path,
                img_size=img_size,
                model_runs=model_runs_for_gc,
                display_w=260,
            )

    st.markdown("---")
    st.markdown("#### Confusion matrix (cached inference)")
    pred_run_keys = [c.replace("_pred", "") for c in df.columns if c.endswith("_pred")]
    if not pred_run_keys:
        st.info("No model prediction columns — train missing fold weights first.")
        return
    cm_options = [(_registry_label_for_run(registry, rk), rk) for rk in pred_run_keys]
    if not cm_options:
        return
    cm_idx = st.selectbox(
        "Model",
        range(len(cm_options)),
        format_func=lambda i: cm_options[i][0],
        key="dc_cm_model",
    )
    cm_label, run_key = cm_options[cm_idx]
    pred_col = f"{run_key}_pred"
    if pred_col in df.columns:
        from sklearn.metrics import confusion_matrix

        y_true = df["true_label"].astype(int).values
        y_pred = df[pred_col].astype(int).values
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{cm_label} — fold {fold_id}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        _show_fig(fig, 380)


def _presets_for_spec(preset_data: Dict[str, Any], spec_id: str) -> List[Dict[str, Any]]:
    return [p for p in preset_data.get("presets", []) if p.get("spec_id") == spec_id]


def _load_preset_into_params(
    spec_id: str,
    preset_name: str,
    preset_data: Dict[str, Any],
) -> Dict[str, Any]:
    spec = catalog()[spec_id]
    base = dict(spec.default_params)
    if preset_name and preset_name != "— custom —":
        match = [p for p in _presets_for_spec(preset_data, spec_id) if p["name"] == preset_name]
        if match:
            base = {**base, **match[0].get("params", {})}
    return base


def _registry_model_run_picker(
    registry: List[Tuple[str, str]],
    params: Dict[str, Any],
    *,
    key_prefix: str,
    require_compare: bool = True,
    root: Optional[Path] = None,
    fold_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Fill baseline_label and model_runs from sidebar registry."""
    out = dict(params)
    if not registry:
        st.warning(
            "Sidebar **Run registry** is empty. Add one line per run, e.g. `reward=reward_percentile_dark`. "
            "Discovered folders under `runs/` are merged automatically."
        )
        return out

    st.caption(
        "Models come from the sidebar registry (`label=folder` under `runs/`). "
        "**Baseline** is one model; **Compare runs** is a multiselect — pick as many as you need (not limited to two)."
    )

    reg_opts = [f"{lab} ({run})" for lab, run in registry]
    baseline_default = str(out.get("baseline_label", registry[0][0]))
    baseline_labels = [lab for lab, _ in registry]
    baseline_idx = baseline_labels.index(baseline_default) if baseline_default in baseline_labels else 0

    st.markdown("##### Models to compare")
    baseline_opt = st.selectbox(
        "Baseline (reference for deltas)",
        reg_opts,
        index=baseline_idx,
        key=f"{key_prefix}_baseline",
    )
    base_lab, _ = baseline_opt.split(" (", 1)
    base_lab = base_lab.strip()
    out["baseline_label"] = base_lab

    compare_opts_all = [opt for opt in reg_opts if not opt.startswith(f"{base_lab} (")]
    compare_default: List[str] = []
    for part in str(out.get("model_runs", "")).split(","):
        part = part.strip()
        if "=" in part:
            lbl, _ = part.split("=", 1)
            if lbl.strip() != base_lab:
                compare_default.append(lbl.strip())
    default_compare = [
        opt for opt in compare_opts_all if opt.split(" (", 1)[0].strip() in compare_default
    ]
    if not default_compare:
        default_compare = list(compare_opts_all)

    if root is not None and fold_id is not None:
        status = []
        for lab, run in registry:
            h5 = model_h5_path(root, run, int(fold_id))
            status.append({"label": lab, "run_folder": run, f"fold_{fold_id}_model": h5.is_file()})
        with st.expander(f"Fold {fold_id} model files on disk", expanded=False):
            st_dataframe(_df(status))

    if require_compare:
        compare_opts = st.multiselect(
            "Compare runs (one or more)",
            options=compare_opts_all,
            default=default_compare,
            key=f"{key_prefix}_compare",
            help="Select every run you want in this job. Preset `model_runs` only sets defaults if labels match.",
        )
    else:
        compare_opts = default_compare

    picks = [baseline_opt] + list(compare_opts)
    parsed: List[Tuple[str, str]] = []
    for opt in picks:
        lab, run = opt.split(" (", 1)
        parsed.append((lab.strip(), run.rstrip(")").strip()))

    if parsed:
        out["model_runs"] = ",".join([f"{lab}={run}" for lab, run in parsed])
        st.caption(f"`model_runs` → `{out['model_runs']}`")
    return out


def _render_param_form(
    spec_id: str,
    params: Dict[str, Any],
    key_prefix: str,
    *,
    hide_fields: Optional[set] = None,
) -> Dict[str, Any]:
    spec = catalog()[spec_id]
    out = dict(params)
    hide_fields = hide_fields or set()
    for name, schema in spec.param_schema.items():
        if name in hide_fields:
            continue
        val = params.get(name, spec.default_params.get(name))
        ptype = schema.get("type", "text")
        label = schema.get("label", name)
        k = f"{key_prefix}_{spec_id}_{name}"
        if ptype == "int":
            out[name] = int(
                st.number_input(
                    label,
                    min_value=int(schema.get("min", 0)),
                    max_value=int(schema.get("max", 1000)),
                    value=int(val) if val is not None else 0,
                    key=k,
                )
            )
        elif ptype == "float":
            out[name] = float(
                st.number_input(
                    label,
                    min_value=float(schema.get("min", 0.0)),
                    max_value=float(schema.get("max", 1.0)),
                    value=float(val) if val is not None else 0.0,
                    step=float(schema.get("step", 0.05)),
                    key=k,
                )
            )
        elif ptype == "bool":
            out[name] = bool(st.checkbox(label, value=bool(val), key=k))
        elif ptype == "select":
            opts = schema.get("options", [])
            idx = opts.index(val) if val in opts else 0
            out[name] = st.selectbox(label, opts, index=idx, key=k)
        else:
            out[name] = st.text_input(label, value=str(val) if val is not None else "", key=k)
    return out


def _load_named_preset(
    preset_data: Dict[str, Any],
    preset_name: str,
    default_spec_id: str,
) -> Tuple[str, Dict[str, Any]]:
    if preset_name and preset_name != "— custom —":
        for p in preset_data.get("presets", []):
            if p.get("name") == preset_name:
                sid = str(p.get("spec_id", default_spec_id))
                return sid, {**catalog()[sid].default_params, **p.get("params", {})}
    return default_spec_id, dict(catalog()[default_spec_id].default_params)


def _experiments_run_actions(
    root: Path,
    spec_id: str,
    params: Dict[str, Any],
    preset_data: Dict[str, Any],
    *,
    preset_name_key: str,
    dry_key: str,
    run_key: str,
    save_key: str,
) -> None:
    spec = catalog()[spec_id]
    try:
        cmd, _ = build_command(root, spec_id, params)
        st.code(command_preview(cmd), language="bash")
    except Exception as e:
        st.error(str(e))
        cmd = None

    preset_name_new = st.text_input(
        "Save as preset (name)",
        value=f"{spec.title} — custom",
        key=preset_name_key,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        dry = st.button("Dry-run (log command only)", key=dry_key)
    with c2:
        run = st.button("Start job", type="primary", key=run_key)
    with c3:
        save_p = st.button("Save preset", key=save_key)

    if save_p and preset_name_new.strip():
        preset_data.setdefault("presets", []).append(
            {"name": preset_name_new.strip(), "spec_id": spec_id, "params": params}
        )
        save_presets(root, preset_data)
        st.success(f"Saved to {presets_path(root)}")
        st_rerun()

    if dry and cmd is not None:
        job = start_job(root, spec_id, params, dry_run=True)
        st.success(f"Dry-run logged: {job['id']}")

    if run and cmd is not None:
        st.warning("Long jobs block until finished. For multi-hour CV, prefer terminal or run overnight.")
        with st.spinner("Running… (see log below)"):
            job = start_job(root, spec_id, params, dry_run=False)
            proc = job.pop("_proc", None)
            if proc is not None:
                rc = proc.wait()
                job["returncode"] = rc
                job["status"] = "success" if rc == 0 else "failed"
        st.session_state["last_job_id"] = job["id"]
        if job.get("status") == "success":
            st.success(f"Finished: {job['id']}")
        else:
            st.error(f"Failed (code {job.get('returncode')}): {job['id']}")


def _tab_experiments(root: Path) -> None:
    st.subheader("Experiment launcher")
    st.caption(
        "Each tab is one pipeline step. Presets are listed only for that step. "
        f"File: `{presets_path(root)}` · Logs: `experiments/logs/`"
    )
    st.markdown(
        "**Suggested order:** Compute weights → CV training → Model comparison / Overlap → Analysis tabs"
    )

    preset_data = load_presets(root)
    registry: List[Tuple[str, str]] = st.session_state.get("registry", [])
    specs = catalog()

    tab_w, tab_train, tab_mc, tab_ov, tab_jobs = st.tabs(
        [
            "1 · Compute weights",
            "2 · CV training",
            "3 · Model comparison",
            "4 · Overlap",
            "5 · Job history",
        ]
    )

    with tab_w:
        st.markdown("### Compute sample weights (GradCAM)")
        st.caption(specs["compute_weights"].description)
        w_presets = _presets_for_spec(preset_data, "compute_weights") + _presets_for_spec(
            preset_data, "compute_density_gap_weights"
        )
        w_names = ["— custom —"] + [p["name"] for p in w_presets]
        w_pick = st.selectbox("Load preset (weights only)", w_names, key="exp_preset_w")
        w_spec, params_w = _load_named_preset(preset_data, w_pick, "compute_weights")
        formula_opts = ["autoopt", "density_gap"]
        wf = str(params_w.get("weight_formula", "autoopt"))
        params_w = _render_param_form(w_spec, params_w, "expw", hide_fields={"weight_formula"})
        params_w["weight_formula"] = st.selectbox(
            "Weight formula",
            formula_opts,
            index=formula_opts.index(wf) if wf in formula_opts else 0,
            key="expw_formula",
        )
        _experiments_run_actions(
            root,
            w_spec,
            params_w,
            preset_data,
            preset_name_key="exp_new_preset_w",
            dry_key="exp_dry_w",
            run_key="exp_run_w",
            save_key="exp_save_w",
        )

    with tab_train:
        st.markdown("### Cross-validation training")
        st.caption(specs["cv_train"].description)
        t_presets = _presets_for_spec(preset_data, "cv_train")
        t_names = ["— custom —"] + [p["name"] for p in t_presets]
        t_pick = st.selectbox("Load preset (training only)", t_names, key="exp_preset_train")
        _, params_t = _load_named_preset(preset_data, t_pick, "cv_train")
        train_hide: set = set()
        if registry:
            run_opts = [f"{lab} ({run})" for lab, run in registry]
            default_run = str(params_t.get("run_name", registry[0][1]))
            idx = next((i for i, (_, r) in enumerate(registry) if r == default_run), 0)
            pick = st.selectbox(
                "Training run (`runs/<name>/` output folder)",
                run_opts,
                index=idx,
                key="exp_train_run",
            )
            params_t["run_name"] = pick.split(" (", 1)[1].rstrip(")")
            wdirs = discover_weights_dirs(root)
            wd_opts = [""] + [str(p.relative_to(root)).replace("\\", "/") for p in wdirs]
            cur_wd = str(params_t.get("weights_dir", "")).replace("\\", "/")
            wd_idx = wd_opts.index(cur_wd) if cur_wd in wd_opts else 0
            params_t["weights_dir"] = st.selectbox(
                "Weights directory",
                wd_opts,
                index=wd_idx,
                format_func=lambda x: "(no sample weights)" if x == "" else x,
                key="exp_train_wd",
            )
            train_hide = {"run_name", "weights_dir"}
        params_t = _render_param_form("cv_train", params_t, "exptr", hide_fields=train_hide)
        _experiments_run_actions(
            root,
            "cv_train",
            params_t,
            preset_data,
            preset_name_key="exp_new_preset_train",
            dry_key="exp_dry_train",
            run_key="exp_run_train",
            save_key="exp_save_train",
        )

    with tab_mc:
        st.markdown("### Model comparison")
        st.caption(specs["model_comparison"].description)
        st.caption("Output: `artifacts/model_comparison/<tag>/` (`fold_metrics.jsonl`, `plots/`, `run_info.json`)")
        mc_presets = _presets_for_spec(preset_data, "model_comparison")
        mc_names = ["— custom —"] + [p["name"] for p in mc_presets]
        mc_pick = st.selectbox("Load preset (model comparison only)", mc_names, key="exp_preset_mc")
        _, params_mc = _load_named_preset(preset_data, mc_pick, "model_comparison")
        params_mc = _registry_model_run_picker(
            registry,
            params_mc,
            key_prefix="expmc",
            root=root,
            fold_id=int(st.session_state.get("default_fold_id", 1)),
        )
        st.caption(
            "**Focus metric:** `density_gap` = inside−outside (no +1, recommended for new-formulation runs). "
            "`density_gap_shifted` = same as training weights (1+inside−outside). "
            "`inside_density` = inside only. Use one metric per comparison job; pick a unique **output tag**."
        )
        params_mc = _render_param_form(
            "model_comparison",
            params_mc,
            "expmc",
            hide_fields={"baseline_label", "model_runs"},
        )
        _experiments_run_actions(
            root,
            "model_comparison",
            params_mc,
            preset_data,
            preset_name_key="exp_new_preset_mc",
            dry_key="exp_dry_mc",
            run_key="exp_run_mc",
            save_key="exp_save_mc",
        )

    with tab_ov:
        st.markdown("### Overlap (focus vs accuracy)")
        st.caption(specs["overlap"].description)
        ov_presets = _presets_for_spec(preset_data, "overlap")
        ov_names = ["— custom —"] + [p["name"] for p in ov_presets]
        ov_pick = st.selectbox("Load preset (overlap only)", ov_names, key="exp_preset_ov")
        _, params_ov = _load_named_preset(preset_data, ov_pick, "overlap")
        if "baseline_label" not in params_ov and registry:
            params_ov["baseline_label"] = registry[0][0]
        params_ov = _registry_model_run_picker(
            registry,
            params_ov,
            key_prefix="expov",
            root=root,
            fold_id=int(st.session_state.get("default_fold_id", 1)),
        )
        st.caption(
            "For density-gap-trained models set **Focus metric** to `density_gap`. "
            "Use a unique **output tag** per overlap run."
        )
        params_ov = _render_param_form(
            "overlap",
            params_ov,
            "expov",
            hide_fields={"baseline_label", "model_runs"},
        )
        out_tag = str(params_ov.get("output_tag", "overlap")).strip() or "overlap"
        st.info(f"Output folder: `artifacts/overlap_accuracy_comparison/{out_tag}/`")
        _experiments_run_actions(
            root,
            "overlap",
            params_ov,
            preset_data,
            preset_name_key="exp_new_preset_ov",
            dry_key="exp_dry_ov",
            run_key="exp_run_ov",
            save_key="exp_save_ov",
        )

    with tab_jobs:
        st.markdown("### Job history")
        jobs = load_jobs(root)
        if not jobs:
            st.info("No jobs yet.")
            return

        if pd is not None:
            st_dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": j["id"],
                            "spec": j["spec_id"],
                            "status": j.get("status"),
                            "returncode": j.get("returncode"),
                            "started": j.get("started_at"),
                        }
                        for j in jobs[:15]
                    ]
                ),
            )

        job_ids = [j["id"] for j in jobs]
        sel = st.selectbox("View log", job_ids, key="exp_log_sel")
        job = next(j for j in jobs if j["id"] == sel)
        st.caption(job.get("command_preview", ""))
        log_text = tail_log(job.get("log_path", ""), n_lines=120)
        st.text_area("Log tail", log_text, height=320, key="exp_log_tail")


def _tab_overlap_artifacts(root: Path) -> None:
    st.subheader("Precomputed overlap artifacts")
    art_root = root / "artifacts" / "overlap_accuracy_comparison"
    if not art_root.is_dir():
        st.info(
            "No overlap artifacts yet. Use the **Experiments** tab or:\n\n"
            "`python scripts/run/overlap_comparison.py --output-tag default`"
        )
        return
    tags = sorted([p.name for p in art_root.iterdir() if p.is_dir()])
    tag = st.selectbox("Artifact tag", tags, key="ov_tag")
    fold_id = int(st.number_input("Fold", 1, 20, 1, key="ov_fold"))
    summ = load_overlap_summary_if_exists(root, tag, fold_id)
    if summ:
        st.json(summ)
    else:
        st.warning("No summary.json for this fold/tag.")

    fold_dir = art_root / tag / f"fold_{fold_id}"
    csv_files = sorted(fold_dir.glob("*_vs_*.csv"))
    if not csv_files:
        csv_files = sorted(fold_dir.glob("baseline_vs_*.csv"))
    for csv_p in csv_files:
        if csv_p.is_file() and pd is not None:
            with st.expander(csv_p.name):
                st_dataframe(pd.read_csv(csv_p).head(200))

    grad_dir = fold_dir / "gradcam_samples"
    if grad_dir.is_dir():
        st.markdown("#### GradCAM sample folders")
        for sub in sorted(grad_dir.iterdir()):
            if sub.is_dir():
                imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
                st.caption(f"{sub.name} ({len(imgs)} images)")
                if imgs:
                    st.image(str(imgs[0]), width=280)


def _tab_guide(root: Path) -> None:
    """In-app guide: Streamlit flow + sample-weight formulas."""
    readme_path = PROJECT_ROOT / "docs" / "STREAMLIT_ANALYSIS_README.md"
    st.subheader("How this Streamlit app works")
    st.caption(
        "Full step-by-step guide (Turkish + English): "
        f"`{readme_path.relative_to(PROJECT_ROOT)}`"
    )
    if readme_path.is_file():
        with st.expander("Open full README in app", expanded=False):
            st.markdown(readme_path.read_text(encoding="utf-8"))

    st.markdown("---")
    st.markdown("### Execution flow (short)")
    st.markdown(
        """
1. You change a widget (fold, button, registry text) in the browser.
2. Streamlit **reruns the whole script** from top to bottom (`main()` → tabs).
3. **Sidebar** reads paths and builds the **run registry** (label → folder under `runs/`).
4. Each **tab** calls a `_tab_*` function; heavy work runs only when you click a button.
5. **`@st.cache_data`** functions (e.g. val inference) reuse results until inputs change.
6. **`st.session_state`** keeps inference tables between reruns so tabs do not recompute every time.
        """
    )
    st.markdown("### Key files")
    st.markdown(
        """
| File | Role |
|------|------|
| `streamlit_ui/analysis_app.py` | UI, tabs, buttons, plots |
| `streamlit_ui/analysis_core.py` | Load runs, weights, val inference, focus comparison |
| `streamlit_ui/helpers.py` | Plots, image helpers, Streamlit version shims |
| `scripts/auto_optimize_gradcam_weights.py` | Focus ratios + weight formulas |
        """
    )

    st.markdown("---")
    st.subheader("Sample weight formulas (reward mode)")
    st.markdown(
        "Weights are computed on the **train split** of each fold, using that fold’s **baseline** model "
        "(see `scripts/compute_fold_weights.py` / Experiments tab)."
    )

    st.markdown("#### 1. GradCAM heatmap (per image, true class)")
    st.latex(r"h = \mathrm{GradCAM}(\text{image},\, y_{\mathrm{true}})")
    st.latex(r"h \leftarrow \frac{h}{\max(h)}")
    st.caption("Heatmap resized to image size, then max-normalized.")

    st.markdown("#### 2. Landmark mask")
    st.markdown(
        "Landmarks → small boxes on eyes/mouth. Background pixels in the mask get "
        "`background_mask_value` (**0** in `weights_percentile/`, often **0.2** in older `weights/`)."
    )

    st.markdown("#### 3. Focus metrics (analysis & comparison)")
    st.markdown(
        """
| ID | Formula | Heatmap | Use when |
|----|---------|---------|----------|
| `focus_ratio` | Σ(h·m) / Σ(h) | max-normalized | autoopt / percentile runs |
| `density_gap` | inside_density − outside_density | clip ≥ 0 only | comparing density-gap-trained models |
| `density_gap_shifted` | 1 + inside − outside | clip ≥ 0 only | same scale as training weights |
| `inside_density` | inside energy / ROI area | clip ≥ 0 only | inside-only training variant |

**Not a bug:** density-gap paths intentionally **do not** max-normalize the heatmap (training uses raw clipped activations).  
Mixing `focus_ratio` on one model and `density_gap` on another in the **same** comparison job is invalid — pick one metric per job.
        """
    )
    st.latex(r"r_{\mathrm{focus}} = \frac{\sum_{i} h_i \, m_i}{\sum_{i} h_i + \varepsilon}")
    st.caption("Classic focus_ratio only (after max-normalize h).")

    st.markdown("#### 4. Reference focus — current (P60)")
    st.latex(r"r_{\mathrm{ref}} = P_{60}(\{r_i\})")
    st.caption("60th percentile of all train focus ratios in that fold (`choose_params` in `auto_optimize_gradcam_weights.py`).")

    st.markdown("#### 4b. Legacy reference — `target_focus` (older `weights/` idea)")
    st.latex(r"\mathrm{offset} = \mathrm{clip}(\sigma_r \times 0.5,\; 0.05,\; 0.15)")
    st.latex(r"r_{\mathrm{target}} = \mathrm{clip}(\mathrm{median}(r) + \mathrm{offset},\; 0.45,\; 0.92)")
    st.markdown(
        """
**What it was meant to do:** build a single “goal” focus level from the fold’s median, with a small data-driven bump, 
but **never below 0.45 or above 0.92**.

**Important:** In today’s script, `target_focus` is still **printed** for logging, but **`r_ref` used in the formula is P60**, not `target_focus`.  
Folders like `weights/` on disk may have been produced by an **older commit** that actually set `r_ref = target_focus`.

**Why weights looked “flat” (≈ 0.93–0.96) with target_focus-style refs:**
- The **0.45 floor** often pushes `r_ref` **above** many samples’ true `r_i` (median focus is often below 0.45).
- Then **most** deltas (r_i minus r_ref) are **negative** → many weights sit near the **lower clip** (similar values).
- With mask **bg = 0.2**, focus ratios are also **numerically smaller** → same effect.
- Net: small spread, curriculum almost uniform.

**Why P60 (`weights_percentile/`) spreads more:**
- `r_ref` **follows the data** (~60% of samples below ref, ~40% above) instead of a fixed high floor.
- More balanced delta signs → alpha search can pick a larger alpha with **higher std** without everything hitting clip.
- **bg = 0** raises measured focus on ROI → larger positive deltas for good samples → wider w range (e.g. ~0.6–2.7 vs ~0.93–1.08).
- **Ranking** of samples (who is up-/down-weighted) often stays similar; mainly **magnitude** changes.
        """
    )

    st.markdown("#### 5. Reward weight")
    st.latex(r"\delta_i = r_i - r_{\mathrm{ref}}")
    st.latex(r"w_i = \mathrm{clip}(1 + \alpha \cdot \delta_i,\; w_{\min},\; w_{\max})")
    st.markdown("If **r_i > r_ref** then **delta > 0** then **w > 1** (up-weight focused samples).")
    st.markdown("If **r_i < r_ref** then **delta < 0** then **w < 1** (down-weight).")
    st.latex(
        r"\alpha \in [0.3,\, 3.0] \text{ chosen by grid search: maximize } "
        r"\mathrm{std}(w) - \lambda_{\mathrm{clip}}\cdot \mathrm{frac\_clipped} "
        r"- \lambda_{\mathrm{mean}}\cdot |\mathrm{mean}(w)-1|"
    )
    st.latex(r"w_{\min} = \max(0.1,\; 1 - 0.8\alpha)")
    st.latex(r"w_{\max} = \min(5,\; 1 + 0.8\alpha)")

    st.markdown("#### 6. Log / exp variants (analysis only — not default CV training)")
    st.latex(r"w_i^{\mathrm{log}} = \log(w_i) - \min_j \log(w_j)")
    st.markdown(
        """
**Log variant:** take natural log of each reward weight, then subtract the minimum so all values are ≥ 0.  
Useful for plots or alternate loss shaping; **not** loaded by default in `cv_main.py`.
        """
    )
    st.latex(r"w_i^{\mathrm{exp}} = \exp\bigl(\min(w_i,\, 50)\bigr)")
    st.markdown(
        """
**Exp variant — what `min(w_i, 50)` means:**

1. Start from the **reward** weight w_i (usually between ~0.1 and ~5 after clip).
2. **`min(w_i, 50)`** caps the exponent so `exp` never overflows (e.g. a bug or huge weight cannot produce `exp(100)`).
3. **`exp(...)`** applies the exponential: differences in w_i become **multiplicative** and much wider.

Examples (reward → exp):

| reward w_i | exp weight |
|----------------|------------|
| 0.9 | ≈ 2.46 |
| 1.0 | ≈ 2.72 |
| 1.2 | ≈ 3.32 |
| 2.0 | ≈ 7.39 |

    st.markdown("#### 6. Log / exp variants (analysis only)")
    st.latex(r"w_i^{\mathrm{log}} = \log(w_i) - \min_j \log(w_j)")
    st.latex(r"w_i^{\mathrm{exp}} = \exp(\min(w_i,\, 50))")
    st.caption("Training CV uses **`fold_k_weights.json`** (reward) unless you change `weights_dir` in the run config.")
""")
    st.markdown("#### 7. Training usage")
    st.markdown(
        "Sample weights go into Keras as `sample_weight` on the **train** generator (`cv_main.py`). "
        "Validation is unweighted. Baseline has no `weights_dir` → implicit weight 1.0."
    )

    wdirs = discover_weights_dirs(root)
    if wdirs:
        st.markdown("#### Weight directories on disk")
        for p in wdirs:
            try:
                st.code(str(p.resolve().relative_to(root.resolve())))
            except ValueError:
                st.code(str(p))


def main() -> None:
    st.set_page_config(page_title="DDD Analysis", layout="wide", initial_sidebar_state="expanded")
    st.title("DDD experiment analysis")
    st.caption("Runs · weights · focus · decisions · training · experiment launcher")

    _tf_init_memory_growth()
    sidebar = st.sidebar
    root = Path(sidebar.text_input("Project root", str(PROJECT_ROOT))).resolve()
    fold_ds = Path(sidebar.text_input("Fold JSON dir", "fold_datasets"))
    if not fold_ds.is_absolute():
        fold_ds = root / fold_ds
    dataset_dir = Path(sidebar.text_input("Dataset dir", "dataset"))
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    img_h = int(sidebar.number_input("Image H", 64, 512, 224))
    img_w = int(sidebar.number_input("Image W", 64, 512, 224))
    registry = _registry_from_sidebar(sidebar, root)
    st.session_state["registry"] = registry
    default_fold = int(sidebar.number_input("Default fold", 1, 20, 1, key="sb_default_fold"))
    st.session_state["default_fold_id"] = default_fold
    fm_opts = list(FOCUS_METRIC_LABELS.keys())
    cur_fm = str(st.session_state.get("focus_metric", "focus_ratio"))
    st.session_state["focus_metric"] = sidebar.selectbox(
        "Live focus metric (GradCAM tab)",
        fm_opts,
        index=fm_opts.index(cur_fm) if cur_fm in fm_opts else 0,
        format_func=lambda k: f"{k}",
        key="sb_focus_metric",
    )
    sidebar.caption(FOCUS_METRIC_LABELS.get(st.session_state["focus_metric"], ""))

    tabs = st.tabs(
        [
            "Guide",
            "Overview",
            "GradCAM & mask",
            "Training",
            "Val reports & CM",
            "Weights",
            "Focus & mask BG",
            "Decisions",
            "Overlap artifacts",
            "Model comparison artifacts",
            "Experiments",
        ]
    )

    with tabs[0]:
        _tab_guide(root)
    with tabs[1]:
        _tab_overview(root, registry)
    with tabs[2]:
        _tab_gradcam_and_mask(
            root, fold_ds, dataset_dir, (img_h, img_w), registry,
            default_fold_id=int(st.session_state.get("default_fold_id", 1)),
        )
    with tabs[3]:
        _tab_training(root, registry)
    with tabs[4]:
        _tab_val_reports(root, registry)
    with tabs[5]:
        _tab_weights(root, fold_ds, dataset_dir, registry, (img_h, img_w))
    with tabs[6]:
        _tab_focus_mask(root, fold_ds, dataset_dir, (img_h, img_w), registry)
    with tabs[7]:
        _tab_decisions(root, fold_ds, dataset_dir, registry, (img_h, img_w))
    with tabs[8]:
        _tab_overlap_artifacts(root)
    with tabs[9]:
        _tab_model_comparison_artifacts(root)
    with tabs[10]:
        _tab_experiments(root)


if __name__ == "__main__":
    main()
