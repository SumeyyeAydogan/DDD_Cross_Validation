"""
DDD cross-validation experiment viewer (Streamlit).

  streamlit run streamlit_app.py

Modules:
  - ROI mask preview
  - GradCAM comparison (pick N models via dropdowns)
  - Training metrics comparison (one fold × many runs, or one run × many folds)
"""
from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import streamlit as st

from src.cv_dataloader import labels_from_paths
from src.fold_functions import load_fold_manifest
from src.focus_metrics import compute_focus_ratio
from src.gradcam import CustomGradCAM
from src.mask_helpers import create_landmark_mask
from streamlit_ui.helpers import (
    discover_run_names,
    fold_run_dir,
    list_folds_for_run,
    load_metrics_csv,
    merge_registry,
    model_h5_path,
    parse_run_registry,
    plot_extra_metrics_fig,
    plot_training_history_fig,
    saved_plot_path,
)

_CLASS_NAMES = ("NotDrowsy", "Drowsy")
_SOURCE_FOLD_VAL = "Fold validation list"
_SOURCE_UPLOAD = "Upload file"

_METRICS_MODE_FOLD = "One fold — compare multiple runs"
_METRICS_MODE_RUN = "One run — compare multiple folds"

_PLOT_HISTORY = "Training history (accuracy & loss)"
_PLOT_EXTRA = "Training metrics (precision, recall, AUC)"
_PLOT_SAVED = "Saved PNG (from disk)"


def _tf_init_memory_growth() -> None:
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


@st.cache_resource(max_entries=16)
def _load_model_weights(h5_abs: str):
    import tensorflow as tf

    return tf.keras.models.load_model(h5_abs, compile=False)


def _load_image_uint8(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(img_size)
    return np.asarray(img).astype(np.uint8)


def _load_upload_uint8(data: bytes, img_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB").resize(img_size)
    return np.asarray(img).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    m = np.clip(mask, 0.0, 1.0)
    h, w = m.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 1] = m
    return (rgb * 255.0).astype(np.uint8)


def _overlay_mask(image_uint8: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = image_uint8.astype(np.float32) / 255.0
    m = np.clip(mask[..., None], 0.0, 1.0)
    tint = np.zeros_like(img)
    tint[..., 1] = m[..., 0]
    out = (1.0 - alpha) * img + alpha * tint
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _truth_from_path(image_path: Optional[str]) -> Optional[int]:
    if not image_path:
        return None
    try:
        return int(labels_from_paths([image_path], _CLASS_NAMES)[0])
    except Exception:
        return None


def _registry_options(registry: List[Tuple[str, str]]) -> List[str]:
    return [f"{lab} ({run})" for lab, run in registry]


def _registry_pick(registry: List[Tuple[str, str]], option: str) -> Tuple[str, str]:
    idx = _registry_options(registry).index(option)
    return registry[idx]


def _sidebar_globals() -> Tuple[Path, Path, int, Tuple[int, int], List[Tuple[str, str]]]:
    sidebar = st.sidebar
    sidebar.header("Project")
    root = Path(sidebar.text_input("Project root", value=str(PROJECT_ROOT))).resolve()
    fold_ds = sidebar.text_input("Fold JSON directory", value="fold_datasets")
    fold_ds_abs = Path(fold_ds) if Path(fold_ds).is_absolute() else (root / fold_ds)

    sidebar.header("Model registry")
    sidebar.caption("One line per model: `label=runs_folder_name`")
    default_map = "baseline=baseline\nreward=reward\nlog=log"
    run_text = sidebar.text_area("Mappings", value=default_map, height=100)
    discovered = discover_run_names(root / "runs")
    registry = merge_registry(parse_run_registry(run_text), discovered)
    if discovered:
        sidebar.caption(f"Auto-discovered under `runs/`: {', '.join(discovered)}")

    sidebar.header("Inference image size")
    img_h = int(sidebar.number_input("Height (px)", value=224, min_value=64, max_value=512))
    img_w = int(sidebar.number_input("Width (px)", value=224, min_value=64, max_value=512))
    fold_id = int(sidebar.number_input("Default fold (1-based)", min_value=1, max_value=20, value=1))

    return root, fold_ds_abs, fold_id, (img_h, img_w), registry


def _pick_models_ui(
    registry: List[Tuple[str, str]],
    *,
    key_prefix: str,
    max_models: int = 6,
) -> List[Tuple[str, str]]:
    if not registry:
        st.error("No models in registry. Add `label=run` lines in the sidebar.")
        return []

    n_avail = len(registry)
    n_compare = int(
        st.selectbox(
            "Number of models to compare",
            options=list(range(1, min(max_models, n_avail) + 1)),
            index=min(1, n_avail - 1),
            key=f"{key_prefix}_n_compare",
        )
    )
    opts = _registry_options(registry)
    picks: List[Tuple[str, str]] = []
    used: set = set()
    cols = st.columns(min(n_compare, 3))
    for i in range(n_compare):
        col = cols[i % len(cols)]
        with col:
            default_idx = min(i, len(opts) - 1)
            choice = st.selectbox(
                f"Model {i + 1}",
                options=opts,
                index=default_idx,
                key=f"{key_prefix}_model_{i}",
            )
        lab, run = _registry_pick(registry, choice)
        if run in used:
            st.warning(f"`{lab}` uses run `{run}` again (duplicate).")
        used.add(run)
        picks.append((lab, run))
    return picks


def _load_image_block(
    fold_ds_abs: Path,
    fold_id: int,
    img_size: Tuple[int, int],
    *,
    key_prefix: str,
) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, float]]:
    use_fold = st.radio(
        "Image source",
        (_SOURCE_FOLD_VAL, _SOURCE_UPLOAD),
        horizontal=True,
        key=f"{key_prefix}_source",
    )
    val_labels_map: Dict[str, float] = {}
    source_path: Optional[str] = None
    image_uint8: Optional[np.ndarray] = None

    if use_fold == _SOURCE_FOLD_VAL:
        json_path = fold_ds_abs / f"fold_{fold_id}.json"
        if not json_path.is_file():
            st.error(f"Fold manifest not found: {json_path}")
            return None, None, {}
        data = load_fold_manifest(fold_id - 1, str(fold_ds_abs))
        val_paths = [str(Path(p)) for p in data["val"]["files"]]
        val_labels = list(data["val"]["labels"])
        val_labels_map = {str(Path(p).resolve()): float(l) for p, l in zip(val_paths, val_labels)}
        if not val_paths:
            st.warning("Validation file list is empty.")
            return None, None, {}
        labels_display = [f"[{i}] {Path(p).name}" for i, p in enumerate(val_paths)]
        choice = st.selectbox(
            "Validation image",
            range(len(val_paths)),
            format_func=lambda i: labels_display[i],
            key=f"{key_prefix}_val_img",
        )
        source_path = str(Path(val_paths[choice]).resolve())
        image_uint8 = _load_image_uint8(source_path, img_size)
    else:
        up = st.file_uploader("PNG / JPG / WebP", type=["png", "jpg", "jpeg", "webp"], key=f"{key_prefix}_up")
        if up is not None:
            image_uint8 = _load_upload_uint8(up.getvalue(), img_size)

    return image_uint8, source_path, val_labels_map


def _mask_preview_block(
    image_uint8: np.ndarray,
    img_size: Tuple[int, int],
    display_width: int,
) -> Optional[np.ndarray]:
    roi_half = int(st.slider("Box half-size (landmark_box_half_size)", 4, 40, 12, key="mask_roi_half"))
    bg_val = float(st.slider("Background value", 0.0, 1.0, 0.2, 0.05, key="mask_bg"))
    mask = create_landmark_mask(
        image_uint8,
        img_size,
        background_mask_value=bg_val,
        landmark_box_half_size=roi_half,
    )
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.caption("Original")
        st.image(image_uint8, width=display_width)
    with mcol2:
        st.caption("Mask")
        if mask is not None:
            st.image(_mask_to_rgb(mask), width=display_width)
        else:
            st.warning("No face detected.")
    with mcol3:
        st.caption("Overlay")
        if mask is not None:
            st.image(_overlay_mask(image_uint8, mask), width=display_width)
        else:
            st.image(image_uint8, width=display_width)
    return mask


def _gradcam_tab(
    root: Path,
    fold_ds_abs: Path,
    default_fold_id: int,
    img_size: Tuple[int, int],
    registry: List[Tuple[str, str]],
) -> None:
    st.subheader("GradCAM comparison")
    fold_id = int(st.number_input("Fold for weights", min_value=1, max_value=20, value=default_fold_id, key="gc_fold"))
    display_w = int(st.slider("GradCAM display width (px)", 120, 600, 260, key="gc_display_w"))
    mask_w = int(st.slider("Mask preview width (px)", 120, 400, 200, key="gc_mask_w"))

    image_uint8, source_path, val_labels_map = _load_image_block(
        fold_ds_abs, fold_id, img_size, key_prefix="gc"
    )
    if image_uint8 is None:
        st.info("Select or upload an image to continue.")
        return

    mask = _mask_preview_block(image_uint8, img_size, mask_w)

    st.markdown("---")
    picks = _pick_models_ui(registry, key_prefix="gc", max_models=6)
    if not picks:
        return

    true_idx: Optional[int] = None
    if source_path and source_path in val_labels_map:
        true_idx = int(val_labels_map[source_path])
    else:
        true_idx = _truth_from_path(source_path)

    if st.button("Compute GradCAM", type="primary", key="gc_compute"):
        import tensorflow as tf

        image_norm = image_uint8.astype(np.float32) / 255.0
        cols = st.columns(len(picks))
        for col, (label, run_name) in zip(cols, picks):
            h5 = model_h5_path(root, run_name, fold_id)
            with col:
                st.markdown(f"**{label}**")
                st.caption(f"`{run_name}` · fold {fold_id}")
                if not h5.is_file():
                    st.error(f"Weights missing:\n`{h5}`")
                    continue
                try:
                    model = _load_model_weights(str(h5.resolve()))
                    cam = CustomGradCAM(model)
                    prob = float(model.predict(image_norm[None, ...], verbose=0)[0, 0])
                    pred = 1 if prob >= 0.5 else 0
                    disp_p = prob if pred == 1 else (1.0 - prob)
                    heatmap = cam.compute_heatmap(image_norm, class_idx=pred)
                    heatmap = tf.image.resize(
                        heatmap[..., None],
                        img_size,
                        method="bilinear",
                        antialias=True,
                    ).numpy()[..., 0]
                    overlay = cam.overlay_heatmap(heatmap, image_norm)
                    st.image(
                        (np.clip(overlay, 0, 1) * 255).astype(np.uint8),
                        width=display_w,
                    )
                    gt = ""
                    if true_idx is not None:
                        gt = f" | GT: {_CLASS_NAMES[int(true_idx)]}"
                    st.caption(f"Pred: {_CLASS_NAMES[pred]} ({disp_p:.3f}){gt}")
                    if mask is not None:
                        st.caption(f"Focus: {compute_focus_ratio(heatmap, mask):.4f}")
                    else:
                        st.caption("Focus: —")
                except Exception as e:
                    st.error(f"Error: {e}")


def _show_figure(fig: plt.Figure, width_px: int) -> None:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, width=width_px)


def _render_metric_panel(
    project_root: Path,
    run_name: str,
    fold_id: int,
    label: str,
    plot_kind: str,
    panel_width: int,
) -> None:
    fold_dir = fold_run_dir(project_root, run_name, fold_id)
    title = f"{label} · fold {fold_id}"

    if plot_kind == _PLOT_SAVED:
        png = saved_plot_path(fold_dir, _PLOT_HISTORY) or saved_plot_path(fold_dir, _PLOT_EXTRA)
        if png is None:
            hist_png = fold_dir / "plots" / "training_history.png"
            met_png = fold_dir / "plots" / "training_metrics.png"
            png = hist_png if hist_png.is_file() else met_png
        if png and png.is_file():
            st.image(str(png), width=panel_width, caption=title)
        else:
            st.caption(title)
            st.warning("No saved PNG. Use CSV plot modes or re-run training.")
        return

    hist = load_metrics_csv(fold_dir / "training_metrics.csv")
    if not hist:
        st.caption(title)
        st.warning(f"No `training_metrics.csv` in `{fold_dir}`")
        return

    if plot_kind == _PLOT_HISTORY:
        fig = plot_training_history_fig(hist, title=title)
    else:
        fig = plot_extra_metrics_fig(hist, title=title)

    if fig is None:
        st.caption(title)
        st.warning("Required columns not found in CSV.")
        return

    _show_figure(fig, panel_width)


def _metrics_tab(root: Path, default_fold_id: int, registry: List[Tuple[str, str]]) -> None:
    st.subheader("Training metrics comparison")
    plot_kind = st.selectbox(
        "Plot type",
        (_PLOT_HISTORY, _PLOT_EXTRA, _PLOT_SAVED),
        key="met_plot_kind",
    )
    panel_w = int(st.slider("Plot panel width (px)", 200, 900, 420, key="met_panel_w"))
    mode = st.selectbox(
        "Comparison mode",
        (_METRICS_MODE_FOLD, _METRICS_MODE_RUN),
        key="met_mode",
    )

    if not registry:
        st.error("No models in registry.")
        return

    if mode == _METRICS_MODE_FOLD:
        fold_id = int(st.number_input("Fold", min_value=1, max_value=20, value=default_fold_id, key="met_fold"))
        picks = _pick_models_ui(registry, key_prefix="met_fold", max_models=6)
        if not picks:
            return
        if st.button("Show plots", type="primary", key="met_show_fold"):
            cols = st.columns(len(picks))
            for col, (label, run_name) in zip(cols, picks):
                with col:
                    _render_metric_panel(root, run_name, fold_id, label, plot_kind, panel_w)
    else:
        run_opts = _registry_options(registry)
        run_choice = st.selectbox("Run to compare across folds", options=run_opts, key="met_run_pick")
        label, run_name = _registry_pick(registry, run_choice)
        folds_avail = list_folds_for_run(root, run_name)
        if not folds_avail:
            st.warning(f"No `fold_*` directories under `runs/{run_name}`.")
            return
        n_folds = int(
            st.selectbox(
                "Number of folds to compare",
                options=list(range(1, len(folds_avail) + 1)),
                index=min(len(folds_avail) - 1, 4),
                key="met_n_folds",
            )
        )
        fold_cols = st.columns(min(n_folds, 3))
        selected_folds: List[int] = []
        for i in range(n_folds):
            with fold_cols[i % len(fold_cols)]:
                default_f = folds_avail[min(i, len(folds_avail) - 1)]
                fid = st.selectbox(
                    f"Fold slot {i + 1}",
                    options=folds_avail,
                    index=folds_avail.index(default_f),
                    key=f"met_fold_slot_{i}",
                )
                selected_folds.append(int(fid))

        if st.button("Show plots", type="primary", key="met_show_run"):
            cols = st.columns(len(selected_folds))
            for col, fid in zip(cols, selected_folds):
                with col:
                    _render_metric_panel(root, run_name, fid, f"{label} (f{fid})", plot_kind, panel_w)


def main() -> None:
    st.set_page_config(page_title="DDD Experiment Viewer", layout="wide")
    st.title("DDD experiment viewer")
    st.caption("Compare ROI masks, GradCAM, and training curves across runs and folds.")
    st.info(
        "Full analysis + experiment launcher (runs, weights, decisions, script runner): "
        "`streamlit run streamlit_analysis_app.py` → **Experiments** tab"
    )

    _tf_init_memory_growth()
    root, fold_ds_abs, default_fold_id, img_size, registry = _sidebar_globals()

    tab_mask, tab_gradcam, tab_metrics = st.tabs(
        ["ROI mask", "GradCAM comparison", "Training metrics"]
    )

    with tab_mask:
        st.subheader("ROI mask preview")
        fold_id = int(st.number_input("Fold (for val list)", min_value=1, max_value=20, value=default_fold_id, key="mask_fold"))
        mask_w = int(st.slider("Display width (px)", 120, 400, 220, key="mask_only_w"))
        image_uint8, _, _ = _load_image_block(fold_ds_abs, fold_id, img_size, key_prefix="mask_only")
        if image_uint8 is not None:
            _mask_preview_block(image_uint8, img_size, mask_w)

    with tab_gradcam:
        _gradcam_tab(root, fold_ds_abs, default_fold_id, img_size, registry)

    with tab_metrics:
        _metrics_tab(root, default_fold_id, registry)


if __name__ == "__main__":
    main()