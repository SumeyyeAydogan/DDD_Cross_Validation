"""Analytics core for Streamlit experiment comparison (runs, weights, focus, decisions)."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.cv_dataloader import labels_from_paths, make_tf_dataset_from_paths
from src.focus_metrics import compute_focus_score
from src.fold_functions import load_fold_manifest, sample_weights_for_train_files
from src.gradcam import CustomGradCAM
from src.mask_helpers import create_landmark_mask, image_to_float01_rgb, image_to_uint8_rgb

CLASS_NAMES: Tuple[str, str] = ("NotDrowsy", "Drowsy")


def _load_metrics_csv_simple(csv_path: Path) -> Dict[str, List[float]]:
    if not csv_path.is_file():
        return {}
    hist: Dict[str, List[float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key in ("epoch", "timestamp") or value in (None, ""):
                    continue
                try:
                    hist.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return hist


def discover_weights_dirs(project_root: Path) -> List[Path]:
    root = project_root.resolve()
    candidates = [
        root / "weights",
        root / "weights_percentile",
    ]
    out: List[Path] = []
    for p in candidates:
        if p.is_dir() and any(p.glob("fold_*_weights.json")):
            out.append(p)
    for p in sorted(root.glob("weights*")):
        if p.is_dir() and p not in out and any(p.glob("fold_*_weights.json")):
            out.append(p)
    return out


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def weights_dir_for_run(project_root: Path, run_name: str) -> Optional[Path]:
    cfg = load_run_config(project_root / "runs" / run_name)
    wd = cfg.get("weights_dir")
    if wd:
        p = Path(wd)
        return p if p.is_absolute() else (project_root / p).resolve()
    return None


def load_weights_map(json_path: Path) -> Dict[str, float]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k).replace("\\", "/"): float(v) for k, v in raw.items()}


def rel_path_from_abs(file_path: str, base_dir: str) -> str:
    return os.path.relpath(os.path.abspath(file_path), os.path.abspath(base_dir)).replace("\\", "/")


def weight_records_for_fold(
    *,
    weights_dir: Path,
    fold_id: int,
    fold_datasets_dir: Path,
    dataset_dir: Path,
    split: str = "train",
    weight_file_suffix: str = "_weights.json",
) -> List[Dict[str, Any]]:
    wpath = weights_dir / f"fold_{fold_id}{weight_file_suffix}"
    if not wpath.is_file():
        return []

    wmap = load_weights_map(wpath)
    data = load_fold_manifest(fold_id - 1, str(fold_datasets_dir))
    files = list(data[split]["files"])
    labels = list(data[split]["labels"])
    base = str(dataset_dir.resolve())

    rows: List[Dict[str, Any]] = []
    for fp, lab in zip(files, labels):
        rel = rel_path_from_abs(fp, base)
        rows.append(
            {
                "rel_path": rel,
                "abs_path": str(Path(fp).resolve()),
                "class_idx": int(lab),
                "class_name": CLASS_NAMES[int(lab)],
                "weight": float(wmap.get(rel, 1.0)),
                "weight_missing": rel not in wmap,
            }
        )
    return rows


def parse_evaluation_report(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw": text, "per_class": {}, "accuracy": None, "roc_auc": None, "test_accuracy": None, "test_loss": None}
    if not text.strip():
        return out

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("accuracy"):
            try:
                out["accuracy"] = float(line.split()[-1])
            except ValueError:
                pass
        if "REAL Test accuracy" in line:
            m_acc = re.search(r"Test accuracy:\s*([0-9.]+)", line)
            m_loss = re.search(r"Test loss:\s*([0-9.]+)", line)
            if m_acc:
                out["test_accuracy"] = float(m_acc.group(1))
            if m_loss:
                out["test_loss"] = float(m_loss.group(1))
        for cname in CLASS_NAMES:
            if line.startswith(cname):
                parts = line.split()
                if len(parts) >= 5:
                    out["per_class"][cname] = {
                        "precision": float(parts[1]),
                        "recall": float(parts[2]),
                        "f1": float(parts[3]),
                        "support": int(float(parts[4])),
                    }

    m_auc = re.search(r"=== ROC–AUC Score ===\s*([0-9.eE+-]+)", text)
    if not m_auc:
        m_auc = re.search(r"=== ROC.AUC Score ===\s*([0-9.eE+-]+)", text)
    if m_auc:
        try:
            out["roc_auc"] = float(m_auc.group(1))
        except ValueError:
            pass
    return out


def parse_cv_summary(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("val_accuracy_mean", "val_accuracy_std", "val_auc_mean", "val_auc_std"):
        m = re.search(rf"{key}\s*=\s*([0-9.+-eE]+)", text)
        if m:
            out[key] = float(m.group(1))
    return out


def best_val_from_history(hist: Dict[str, List[float]]) -> Dict[str, float]:
    best: Dict[str, float] = {}
    if "val_accuracy" in hist and hist["val_accuracy"]:
        i = int(np.argmax(hist["val_accuracy"]))
        best["best_epoch"] = float(i + 1)
        best["val_accuracy"] = float(hist["val_accuracy"][i])
        for k in ("val_auc", "val_loss", "val_precision", "val_recall"):
            if k in hist and len(hist[k]) > i:
                best[k] = float(hist[k][i])
    return best


def collect_fold_run_summaries(project_root: Path, run_name: str) -> List[Dict[str, Any]]:
    run_dir = project_root / "runs" / run_name
    rows: List[Dict[str, Any]] = []
    if not run_dir.is_dir():
        return rows
    cfg = load_run_config(run_dir)
    for fold_dir in sorted(run_dir.glob("fold_*")):
        try:
            fold_id = int(fold_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        hist = _load_metrics_csv_simple(fold_dir / "training_metrics.csv")
        best = best_val_from_history(hist)
        report_path = fold_dir / "plots" / "val_evaluation_report.txt"
        report = parse_evaluation_report(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        rows.append(
            {
                "fold": fold_id,
                "epochs": len(hist.get("accuracy", [])),
                "best_val_accuracy": best.get("val_accuracy"),
                "best_val_auc": best.get("val_auc"),
                "report_test_accuracy": report.get("test_accuracy"),
                "report_roc_auc": report.get("roc_auc"),
                "weights_dir": cfg.get("weights_dir"),
                "model_exists": (fold_dir / "models" / f"fold_{fold_id}.h5").is_file(),
            }
        )
    return rows


def compare_accuracy_group(
    baseline_pred: Optional[int],
    compare_pred: Optional[int],
    true_label: int,
) -> str:
    if baseline_pred is None or compare_pred is None:
        return "unknown"
    b_ok = int(baseline_pred == true_label)
    c_ok = int(compare_pred == true_label)
    if c_ok > b_ok:
        return "improved"
    if c_ok < b_ok:
        return "worsened"
    return "same"


def compare_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "no_face"
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "equal"


def model_cache_key(model_path: Path) -> str:
    p = model_path.resolve()
    if not p.is_file():
        return str(p)
    st = p.stat()
    return f"{p}|{st.st_mtime_ns}|{st.st_size}"


def predict_on_paths(
    model_path: str,
    file_paths: Sequence[str],
    img_size: Tuple[int, int],
    *,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    paths = [str(p) for p in file_paths]
    if not paths:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    labels = labels_from_paths(paths, CLASS_NAMES)
    ds = make_tf_dataset_from_paths(
        paths,
        labels,
        img_size,
        batch_size=batch_size,
        augment=False,
        sample_weights=None,
        seed=None,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    probs_list: List[np.ndarray] = []
    for batch in ds:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        preds = model.predict(x, verbose=0).reshape(-1)
        preds = np.clip(np.nan_to_num(preds, nan=0.5), 0.0, 1.0)
        probs_list.append(preds.astype(np.float32))
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=np.float32)
    pred_cls = (probs >= 0.5).astype(np.int32)
    return probs, pred_cls


def confidence_for_display(prob_drowsy: float, pred_class: int) -> float:
    p = float(prob_drowsy)
    return p if pred_class == 1 else (1.0 - p)


def build_val_decision_rows(
    *,
    val_paths: Sequence[str],
    val_labels: Sequence[int],
    model_specs: Dict[str, str],
    dataset_dir: Path,
    weights_map: Optional[Dict[str, float]] = None,
    baseline_key: Optional[str] = None,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    paths = [str(p) for p in val_paths]
    labels = [int(x) for x in val_labels]
    base = str(dataset_dir.resolve())

    preds_by_label: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, mpath in model_specs.items():
        if not mpath or not os.path.isfile(mpath):
            continue
        probs, pred_cls = predict_on_paths(mpath, paths, img_size, batch_size=batch_size)
        preds_by_label[label] = (probs, pred_cls)

    rows: List[Dict[str, Any]] = []
    for i, (fp, y) in enumerate(zip(paths, labels)):
        rel = rel_path_from_abs(fp, base)
        row: Dict[str, Any] = {
            "image_path": fp,
            "rel_path": rel,
            "true_label": y,
            "true_class": CLASS_NAMES[y],
            "weight": float(weights_map.get(rel, np.nan)) if weights_map else np.nan,
        }
        for label, (probs, pred_cls) in preds_by_label.items():
            if i >= len(probs):
                continue
            pr = int(pred_cls[i])
            pb = float(probs[i])
            row[f"{label}_prob"] = pb
            row[f"{label}_pred"] = pr
            row[f"{label}_pred_class"] = CLASS_NAMES[pr]
            row[f"{label}_correct"] = int(pr == y)
            row[f"{label}_confidence"] = confidence_for_display(pb, pr)
        rows.append(row)

    base_key = baseline_key or (next(iter(preds_by_label)) if preds_by_label else None)
    if base_key and base_key in preds_by_label:
        for label in preds_by_label:
            if label == base_key:
                continue
            for row in rows:
                bp = row.get(f"{base_key}_pred")
                cp = row.get(f"{label}_pred")
                row[f"flip_{label}"] = int(bp is not None and cp is not None and bp != cp)
                row[f"acc_delta_{label}"] = compare_accuracy_group(bp, cp, row["true_label"])
    return rows


def summarize_decision_flips(rows: List[Dict[str, Any]], compare_label: str) -> Dict[str, Any]:
    flip_key = f"flip_{compare_label}"
    acc_key = f"acc_delta_{compare_label}"
    n = len(rows)
    flips = [r for r in rows if r.get(flip_key) == 1]
    n_flip = len(flips)
    conf_key = f"{compare_label}_confidence"
    confs = [float(r[conf_key]) for r in flips if conf_key in r and r[conf_key] is not None]
    acc_improved = sum(1 for r in rows if r.get(acc_key) == "improved")
    acc_worsened = sum(1 for r in rows if r.get(acc_key) == "worsened")
    acc_same = sum(1 for r in rows if r.get(acc_key) == "same")
    return {
        "n_total": n,
        "n_flip": n_flip,
        "flip_rate": (n_flip / n) if n else 0.0,
        "acc_improved": acc_improved,
        "acc_worsened": acc_worsened,
        "acc_same": acc_same,
        "flip_confidence_mean": float(np.mean(confs)) if confs else float("nan"),
        "flip_confidence_median": float(np.median(confs)) if confs else float("nan"),
    }


def compute_focus_for_path(
    model_path: str,
    image_path: str,
    img_size: Tuple[int, int],
    background_mask_value: float,
    landmark_box_half_size: int = 12,
    focus_metric: str = "focus_ratio",
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    import tensorflow as tf
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize(img_size)
    image_uint8 = np.asarray(img).astype(np.uint8)
    image_float01 = image_to_float01_rgb(image_uint8)

    model = tf.keras.models.load_model(model_path, compile=False)
    cam = CustomGradCAM(model)
    prob = float(model.predict(image_float01[None, ...], verbose=0)[0][0])
    pred = 1 if prob >= 0.5 else 0

    heatmap = cam.compute_heatmap(image_float01, class_idx=pred)
    heatmap = tf.image.resize(
        heatmap[..., None],
        img_size,
        method="bilinear",
        antialias=True,
    ).numpy()[..., 0]

    mask = create_landmark_mask(
        image_uint8,
        img_size,
        background_mask_value=float(background_mask_value),
        landmark_box_half_size=int(landmark_box_half_size),
    )
    if mask is None:
        return None, pred, prob
    hm = np.clip(heatmap, 0.0, None).astype(np.float32)
    score, _ = compute_focus_score(hm, mask, metric=focus_metric)
    return float(score), pred, prob


def focus_bg_comparison_records(
    *,
    model_path: str,
    image_paths: Sequence[str],
    img_size: Tuple[int, int],
    bg_values: Sequence[float] = (0.0, 0.2),
    dataset_dir: Optional[str] = None,
    landmark_box_half_size: int = 12,
    focus_metric: str = "focus_ratio",
) -> List[Dict[str, Any]]:
    """
    Same trained model per image; only the ROI mask background changes the focus ratio.
    Prediction / probability do not depend on the mask.
    """
    import tensorflow as tf
    from PIL import Image

    model = tf.keras.models.load_model(model_path, compile=False)
    cam = CustomGradCAM(model)
    rows: List[Dict[str, Any]] = []
    base = os.path.abspath(dataset_dir) if dataset_dir else None

    for fp in image_paths:
        rec: Dict[str, Any] = {"image_path": fp}
        if base:
            try:
                rec["rel_path"] = rel_path_from_abs(fp, base)
            except ValueError:
                rec["rel_path"] = fp
        else:
            rec["rel_path"] = fp
        rel = rec["rel_path"].replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        rec["sample"] = f"{parts[-2]} / {parts[-1]}" if len(parts) >= 2 else (parts[-1] if parts else rel)
        try:
            rec["true_class"] = CLASS_NAMES[int(labels_from_paths([fp], CLASS_NAMES)[0])]
        except Exception:
            rec["true_class"] = ""

        img = Image.open(fp).convert("RGB").resize(img_size)
        image_uint8 = np.asarray(img).astype(np.uint8)
        image_float01 = image_to_float01_rgb(image_uint8)
        prob = float(model.predict(image_float01[None, ...], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0
        rec["prob_drowsy"] = prob
        rec["pred"] = pred
        rec["pred_class"] = CLASS_NAMES[pred]

        heatmap = cam.compute_heatmap(image_float01, class_idx=pred)
        heatmap = tf.image.resize(
            heatmap[..., None],
            img_size,
            method="bilinear",
            antialias=True,
        ).numpy()[..., 0]

        for bg in bg_values:
            mask = create_landmark_mask(
                image_uint8,
                img_size,
                background_mask_value=float(bg),
                landmark_box_half_size=int(landmark_box_half_size),
            )
            if mask is not None:
                hm = np.clip(heatmap, 0.0, None).astype(np.float32)
                rec[f"focus_bg_{bg:g}"], _ = compute_focus_score(
                    hm, mask, metric=focus_metric
                )
            else:
                rec[f"focus_bg_{bg:g}"] = None

        f0 = rec.get("focus_bg_0")
        f2 = rec.get("focus_bg_0.2")
        if f0 is not None and f2 is not None:
            rec["focus_delta_0.2_minus_0"] = float(f2) - float(f0)
        rows.append(rec)
    return rows


def runs_with_fold_model(
    root: Path,
    registry: List[Tuple[str, str]],
    fold_id: int,
) -> List[Tuple[str, str, bool, str]]:
    """Return (label, run_name, model_exists, model_path) for sidebar registry."""
    out: List[Tuple[str, str, bool, str]] = []
    for label, run_name in registry:
        h5 = root / "runs" / run_name / f"fold_{fold_id}" / "models" / f"fold_{fold_id}.h5"
        out.append((label, run_name, h5.is_file(), str(h5)))
    return out


def load_overlap_summary_if_exists(project_root: Path, tag: str, fold_id: int) -> Optional[Dict[str, Any]]:
    p = project_root / "artifacts" / "overlap_accuracy_comparison" / tag / f"fold_{fold_id}" / "summary.json"
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def try_load_overlap_detail_csv(project_root: Path, tag: str, fold_id: int) -> List[Dict[str, Any]]:
    import csv

    p = project_root / "artifacts" / "overlap_accuracy_comparison" / tag / f"fold_{fold_id}" / "overlap_details.csv"
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
