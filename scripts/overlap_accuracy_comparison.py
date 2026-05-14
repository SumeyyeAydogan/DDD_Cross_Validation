import csv
import json
import os
import shutil
import sys
import tempfile
import time
import base64
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# Add root path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.focus_metrics import compute_focus_ratio
from src.gradcam import CustomGradCAM
from src.fold_functions import load_fold_manifest
from src.cv_dataloader import labels_from_paths
from src.mask_helpers import create_landmark_mask

_CLASS_NAMES = ("NotDrowsy", "Drowsy")

CONFIG = {
    "output_tag": "",
    "fold_start": 1,
    "fold_count": 5,
    "img_size": (224, 224),
    "background_mask_value": 0.2,
    "roi_padding_px": 12,
    "roi_keep_aspect_pad_x_min_scale": 0.2,
    "save_fold_detail_csv": False,
    "save_gradcam_samples": True,
    "gradcam_samples_per_group": 30,
    "shuffle_gradcam_samples": True,
    "shuffle_seed": 42,
    "save_combined_svg": True,
    "gradcam_png_read_retries": 40,
    "gradcam_png_read_delay_sec": 0.05,
    "gradcam_png_render_retries": 6,
    "save_intermediate_gradcam_images": False,
    "resume_skip_fold_if_summary_exists": True,
    "gradcam_skip_if_combined_exists": True,
    "gradcam_skip_combined_min_bytes": 1024,
    "fold_datasets_dir": "fold_datasets",
    # Tag -> directory name under project_root/runs/ (cv_main: runs/<name>/fold_k/models/fold_k.h5)
    "model_runs": {
        "baseline": "baseline",
        "reward": "reward",
        "log": "log",
        "exp": "exp",
    },
}


def _fold_datasets_abs(root: Path) -> str:
    rel = str(CONFIG.get("fold_datasets_dir", "fold_datasets") or "fold_datasets")
    if os.path.isabs(rel):
        return rel
    return str(root / rel)


def _resolve_fold_paths(fold_id: int) -> Tuple[List[str], Dict[str, str]]:
    fds = _fold_datasets_abs(project_root)
    data = load_fold_manifest(fold_id - 1, fds)
    val_paths = list(data["val"]["files"])

    runs_map: Dict[str, str] = CONFIG.get("model_runs") or {}
    need = ("baseline", "reward", "log", "exp")
    for k in need:
        if k not in runs_map:
            raise KeyError(f"CONFIG['model_runs'] must contain '{k}'")

    model_paths: Dict[str, str] = {}
    for tag in need:
        run_name = str(runs_map[tag])
        h5 = project_root / "runs" / run_name / f"fold_{fold_id}" / "models" / f"fold_{fold_id}.h5"
        model_paths[tag] = str(h5)

    return val_paths, model_paths


def _load_image(path: str, img_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(img_size)
    return np.asarray(img).astype(np.uint8)


def _compute_focus_ratio_for_model(
    *,
    model: tf.keras.Model,
    cam: CustomGradCAM,
    image_uint8: np.ndarray,
    mask: np.ndarray,
    img_size: Tuple[int, int],
) -> Tuple[float, int]:
    image_norm = image_uint8.astype(np.float32) / 255.0
    prob = float(model.predict(image_norm[None, ...], verbose=0)[0][0])
    pred = 1 if prob >= 0.5 else 0

    heatmap = cam.compute_heatmap(image_norm, class_idx=pred)
    heatmap = tf.image.resize(
        heatmap[..., None],
        img_size,
        method="bilinear",
        antialias=True,
    ).numpy()[..., 0]
    return float(compute_focus_ratio(heatmap, mask)), int(pred)


def _safe_float(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{float(v):.8f}"


def _write_details_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold_id",
                "image_path",
                "face_ok",
                "baseline",
                "reward",
                "log",
                "exp",
                "delta_reward",
                "delta_log",
                "delta_exp",
                "group_reward",
                "group_log",
                "group_exp",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_comparison_csv(
    *,
    detail_rows: List[Dict[str, object]],
    fold_dir: Path,
    comp: str,
) -> Dict[str, object]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    output_csv = fold_dir / f"baseline_vs_{comp}.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold_id",
                "image_path",
                "face_ok",
                "baseline",
                comp,
                f"delta_{comp}",
                f"group_{comp}",
            ],
        )
        writer.writeheader()
        for r in detail_rows:
            writer.writerow(
                {
                    "fold_id": r["fold_id"],
                    "image_path": r["image_path"],
                    "face_ok": r["face_ok"],
                    "baseline": r["baseline"],
                    comp: r[comp],
                    f"delta_{comp}": r[f"delta_{comp}"],
                    f"group_{comp}": r[f"group_{comp}"],
                }
            )

    deltas = [float(r[f"delta_{comp}"]) for r in detail_rows if r[f"delta_{comp}"] not in ("", None)]
    summary = {
        "comparison": f"baseline_vs_{comp}",
        "n_total": int(len(detail_rows)),
        "n_face_ok": int(sum(int(r["face_ok"]) for r in detail_rows)),
        "increased": int(sum(1 for d in deltas if d > 0)),
        "decreased": int(sum(1 for d in deltas if d < 0)),
        "equal": int(sum(1 for d in deltas if d == 0)),
        "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "csv_path": str(output_csv),
    }
    with open(fold_dir / f"baseline_vs_{comp}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _compare_delta(delta: Optional[float]) -> str:
    if delta is None:
        return "no_face"
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "equal"


def _compare_accuracy_group(
    baseline_pred: Optional[int],
    compare_pred: Optional[int],
    true_label: int,
) -> str:
    if baseline_pred is None or compare_pred is None:
        return "acc_unknown"
    baseline_correct = int(baseline_pred == true_label)
    compare_correct = int(compare_pred == true_label)
    if compare_correct > baseline_correct:
        return "acc_improved"
    if compare_correct < baseline_correct:
        return "acc_worsened"
    return "acc_same"


def _select_rows_for_group(
    detail_rows: List[Dict[str, object]],
    comp: str,
    group_name: str,
    acc_group_name: str,
    max_count: int,
) -> List[Dict[str, object]]:
    selected = [
        r for r in detail_rows
        if int(r["face_ok"]) == 1
        and r[f"group_{comp}"] == group_name
        and r[f"acc_group_{comp}"] == acc_group_name
        and r[f"delta_{comp}"] not in ("", None)
    ]
    if bool(CONFIG.get("shuffle_gradcam_samples", True)):
        seed_base = int(CONFIG.get("shuffle_seed", 42))
        random.Random(f"{seed_base}:{comp}:{group_name}:{acc_group_name}").shuffle(selected)
    else:
        selected.sort(key=lambda r: abs(float(r[f"delta_{comp}"])), reverse=True)
    return selected[:max_count]


def _pil_open_rgb_robust(path: Path) -> Image.Image:
    """
    Matplotlib savefig + immediate PIL open can race on slow/NFS storage.
    Verify then reopen; retry with backoff on transient read errors.
    """
    retries = int(CONFIG.get("gradcam_png_read_retries", 40))
    delay = float(CONFIG.get("gradcam_png_read_delay_sec", 0.05))
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            if not path.is_file():
                time.sleep(delay)
                continue
            if path.stat().st_size == 0:
                time.sleep(delay)
                continue
            with Image.open(path) as im:
                im.verify()
            img = Image.open(path).convert("RGB")
            img.load()
            return img.copy()
        except (OSError, ValueError) as e:
            last_err = e
            time.sleep(delay * min(1.0 + attempt / 10.0, 5.0))
    raise OSError(f"Could not read PNG after {retries} attempts: {path}") from last_err


def _save_gradcam_png_safe(
    *,
    cam: CustomGradCAM,
    image_norm: np.ndarray,
    true_label: int,
    save_path: Path,
) -> None:
    """
    Render GradCAM to local temp first, verify, then copy to target path.
    Helps avoid corrupted PNG writes on network filesystems.
    """
    retries = int(CONFIG.get("gradcam_png_render_retries", 6))
    delay = float(CONFIG.get("gradcam_png_read_delay_sec", 0.05))
    last_err: Optional[Exception] = None

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.gettempdir()) / "gradcam_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        tmp_png = tmp_dir / f"{save_path.stem}_{os.getpid()}_{attempt}.png"
        try:
            cam.visualize(
                image_norm,
                class_names=("NotDrowsy", "Drowsy"),
                true_idx=int(true_label),
                save_path=str(tmp_png),
            )
            _pil_open_rgb_robust(tmp_png)
            shutil.copyfile(str(tmp_png), str(save_path))
            _pil_open_rgb_robust(save_path)
            try:
                tmp_png.unlink(missing_ok=True)
            except Exception:
                pass
            return
        except Exception as e:
            last_err = e
            time.sleep(delay * min(1.0 + attempt / 5.0, 5.0))
            try:
                tmp_png.unlink(missing_ok=True)
            except Exception:
                pass

    raise RuntimeError(f"Failed to render valid GradCAM PNG: {save_path}") from last_err


def _render_gradcam_temp_png(
    *,
    cam: CustomGradCAM,
    image_norm: np.ndarray,
    true_label: int,
) -> Path:
    """
    Render a GradCAM PNG into temp directory and return temp path.
    """
    retries = int(CONFIG.get("gradcam_png_render_retries", 6))
    delay = float(CONFIG.get("gradcam_png_read_delay_sec", 0.05))
    last_err: Optional[Exception] = None
    tmp_dir = Path(tempfile.gettempdir()) / "gradcam_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        tmp_png = tmp_dir / f"gradcam_{os.getpid()}_{int(time.time()*1000)}_{attempt}.png"
        try:
            cam.visualize(
                image_norm,
                class_names=("NotDrowsy", "Drowsy"),
                true_idx=int(true_label),
                save_path=str(tmp_png),
            )
            _pil_open_rgb_robust(tmp_png)
            return tmp_png
        except Exception as e:
            last_err = e
            time.sleep(delay * min(1.0 + attempt / 5.0, 5.0))
            try:
                tmp_png.unlink(missing_ok=True)
            except Exception:
                pass

    raise RuntimeError("Failed to render temp GradCAM PNG") from last_err


def _combine_vertical_images(image_paths: List[Path], output_path: Path) -> None:
    imgs = [_pil_open_rgb_robust(p) for p in image_paths]
    out_w = max(img.width for img in imgs)
    out_h = sum(img.height for img in imgs)
    out = Image.new("RGB", (out_w, out_h), "white")
    y = 0
    for img in imgs:
        out.paste(img, (0, y))
        y += img.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def _add_top_label(input_path: Path, output_path: Path, label_text: str) -> None:
    img = _pil_open_rgb_robust(input_path)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = draw.textbbox((0, 0), label_text, font=font)
    text_w = x1 - x0
    text_h = y1 - y0

    pad_x, pad_y = 8, 6
    banner_h = text_h + 2 * pad_y
    out = Image.new("RGB", (img.width, img.height + banner_h), "white")
    out.paste(img, (0, banner_h))

    draw_out = ImageDraw.Draw(out)
    draw_out.rectangle([(0, 0), (img.width, banner_h)], fill=(0, 0, 0))
    text_x = max(0, (img.width - text_w) // 2)
    text_y = max(0, (banner_h - text_h) // 2)
    draw_out.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def _save_combined_vertical_svg(image_paths: List[Path], output_svg_path: Path) -> None:
    imgs = [_pil_open_rgb_robust(p) for p in image_paths]
    out_w = max(img.width for img in imgs)
    out_h = sum(img.height for img in imgs)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{out_w}" height="{out_h}" viewBox="0 0 {out_w} {out_h}">'
    ]

    y = 0
    for img in imgs:
        buf = BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        svg_parts.append(
            f'<image x="0" y="{y}" width="{img.width}" height="{img.height}" '
            f'href="data:image/png;base64,{encoded}" />'
        )
        y += img.height

    svg_parts.append("</svg>")
    output_svg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))


def _save_group_gradcam_examples(
    *,
    detail_rows: List[Dict[str, object]],
    fold_dir: Path,
    comp: str,
    models: Dict[str, tf.keras.Model],
    cams: Dict[str, CustomGradCAM],
    img_size: Tuple[int, int],
) -> Dict[str, object]:
    base_dir = fold_dir / "gradcam_samples" / f"baseline_vs_{comp}"
    max_count = int(CONFIG.get("gradcam_samples_per_group", 20))
    result: Dict[str, object] = {"comparison": f"baseline_vs_{comp}", "groups": {}}

    for group_name in ("increased", "decreased"):
        for acc_group_name in ("acc_improved", "acc_worsened", "acc_same"):
            rows = _select_rows_for_group(
                detail_rows,
                comp,
                group_name,
                acc_group_name,
                max_count=max_count,
            )
            group_dir = base_dir / group_name / acc_group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            svg_dir = group_dir / "svg"
            if bool(CONFIG.get("save_combined_svg", True)):
                svg_dir.mkdir(parents=True, exist_ok=True)

            manifest_csv = group_dir / "manifest.csv"
            manifest_txt = group_dir / "manifest.txt"
            saved_count = 0

            with open(manifest_csv, "w", newline="", encoding="utf-8") as f_csv, open(manifest_txt, "w", encoding="utf-8") as f_txt:
                writer = csv.DictWriter(
                    f_csv,
                    fieldnames=[
                        "fold_id",
                        "comparison",
                        "overlap_group",
                        "acc_group",
                        "image_path",
                        "true_label",
                        "baseline_pred",
                        f"{comp}_pred",
                        "delta",
                        "baseline_focus_ratio",
                        f"{comp}_focus_ratio",
                        "baseline_gradcam_path",
                        "compare_gradcam_path",
                        "combined_gradcam_path",
                        "combined_gradcam_svg_path",
                    ],
                )
                writer.writeheader()

                for i, row in enumerate(rows, start=1):
                    image_path = str(row["image_path"])
                    stem = Path(image_path).stem
                    prefix = f"{i:03d}_{stem}"
                    baseline_png = group_dir / f"{prefix}_baseline.png"
                    compare_png = group_dir / f"{prefix}_{comp}.png"
                    baseline_labeled_png = group_dir / f"{prefix}_baseline_labeled.png"
                    compare_labeled_png = group_dir / f"{prefix}_{comp}_labeled.png"
                    combined_png = group_dir / f"{prefix}_combined_vertical.png"
                    combined_svg = svg_dir / f"{prefix}_combined_vertical.svg"

                    skip_gc = bool(CONFIG.get("gradcam_skip_if_combined_exists", False))
                    min_b = int(CONFIG.get("gradcam_skip_combined_min_bytes", 1024))
                    need_svg = bool(CONFIG.get("save_combined_svg", True))
                    combined_ok = combined_png.is_file() and combined_png.stat().st_size >= min_b
                    svg_ok = (not need_svg) or (
                        combined_svg.is_file() and combined_svg.stat().st_size >= 64
                    )
                    if skip_gc and combined_ok and svg_ok:
                        print(f"[RESUME] GradCAM skip (exists): {combined_png}")
                        writer.writerow(
                            {
                                "fold_id": row["fold_id"],
                                "comparison": f"baseline_vs_{comp}",
                                "overlap_group": group_name,
                                "acc_group": acc_group_name,
                                "image_path": image_path,
                                "true_label": row["true_label"],
                                "baseline_pred": row["baseline_pred"],
                                f"{comp}_pred": row[f"{comp}_pred"],
                                "delta": row[f"delta_{comp}"],
                                "baseline_focus_ratio": row["baseline"],
                                f"{comp}_focus_ratio": row[comp],
                                "baseline_gradcam_path": str(baseline_png),
                                "compare_gradcam_path": str(compare_png),
                                "combined_gradcam_path": str(combined_png),
                                "combined_gradcam_svg_path": str(combined_svg)
                                if need_svg
                                else "",
                            }
                        )
                        f_txt.write(f"{image_path}\n")
                        saved_count += 1
                        continue

                    image_uint8 = _load_image(image_path, img_size)
                    image_norm = image_uint8.astype(np.float32) / 255.0

                    tmp_baseline = _render_gradcam_temp_png(
                        cam=cams["baseline"],
                        image_norm=image_norm,
                        true_label=int(row["true_label"]),
                    )
                    tmp_compare = _render_gradcam_temp_png(
                        cam=cams[comp],
                        image_norm=image_norm,
                        true_label=int(row["true_label"]),
                    )
                    tmp_baseline_labeled = tmp_baseline.with_name(tmp_baseline.stem + "_labeled.png")
                    tmp_compare_labeled = tmp_compare.with_name(tmp_compare.stem + "_labeled.png")
                    try:
                        _add_top_label(tmp_baseline, tmp_baseline_labeled, "BASELINE")
                        _add_top_label(tmp_compare, tmp_compare_labeled, comp.upper())
                        _combine_vertical_images([tmp_baseline_labeled, tmp_compare_labeled], combined_png)
                        if bool(CONFIG.get("save_combined_svg", True)):
                            _save_combined_vertical_svg([tmp_baseline_labeled, tmp_compare_labeled], combined_svg)

                        if bool(CONFIG.get("save_intermediate_gradcam_images", False)):
                            shutil.copyfile(str(tmp_baseline), str(baseline_png))
                            shutil.copyfile(str(tmp_compare), str(compare_png))
                            shutil.copyfile(str(tmp_baseline_labeled), str(baseline_labeled_png))
                            shutil.copyfile(str(tmp_compare_labeled), str(compare_labeled_png))
                    finally:
                        for p in (tmp_baseline, tmp_compare, tmp_baseline_labeled, tmp_compare_labeled):
                            try:
                                p.unlink(missing_ok=True)
                            except Exception:
                                pass

                    writer.writerow(
                        {
                            "fold_id": row["fold_id"],
                            "comparison": f"baseline_vs_{comp}",
                            "overlap_group": group_name,
                            "acc_group": acc_group_name,
                            "image_path": image_path,
                            "true_label": row["true_label"],
                            "baseline_pred": row["baseline_pred"],
                            f"{comp}_pred": row[f"{comp}_pred"],
                            "delta": row[f"delta_{comp}"],
                            "baseline_focus_ratio": row["baseline"],
                            f"{comp}_focus_ratio": row[comp],
                            "baseline_gradcam_path": str(baseline_png) if bool(CONFIG.get("save_intermediate_gradcam_images", False)) else "",
                            "compare_gradcam_path": str(compare_png) if bool(CONFIG.get("save_intermediate_gradcam_images", False)) else "",
                            "combined_gradcam_path": str(combined_png),
                            "combined_gradcam_svg_path": str(combined_svg) if bool(CONFIG.get("save_combined_svg", True)) else "",
                        }
                    )
                    f_txt.write(f"{image_path}\n")
                    saved_count += 1

            result["groups"][f"{group_name}/{acc_group_name}"] = {
                "selected_count": int(saved_count),
                "manifest_csv": str(manifest_csv),
                "manifest_txt": str(manifest_txt),
                "output_dir": str(group_dir),
            }
    return result


def _build_fold_summary(
    fold_id: int,
    detail_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "fold_id": int(fold_id),
        "n_total": int(len(detail_rows)),
        "n_face_ok": int(sum(int(r["face_ok"]) for r in detail_rows)),
    }

    for comp in ("reward", "log", "exp"):
        deltas = [float(r[f"delta_{comp}"]) for r in detail_rows if r[f"delta_{comp}"] not in ("", None)]
        summary[f"{comp}_increased"] = int(sum(1 for d in deltas if d > 0))
        summary[f"{comp}_decreased"] = int(sum(1 for d in deltas if d < 0))
        summary[f"{comp}_equal"] = int(sum(1 for d in deltas if d == 0))
        summary[f"{comp}_mean_delta"] = float(np.mean(deltas)) if deltas else float("nan")
        summary[f"{comp}_acc_improved"] = int(sum(1 for r in detail_rows if r.get(f"acc_group_{comp}") == "acc_improved"))
        summary[f"{comp}_acc_worsened"] = int(sum(1 for r in detail_rows if r.get(f"acc_group_{comp}") == "acc_worsened"))
        summary[f"{comp}_acc_same"] = int(sum(1 for r in detail_rows if r.get(f"acc_group_{comp}") == "acc_same"))
        for overlap_group in ("increased", "decreased"):
            for acc_group in ("acc_improved", "acc_worsened", "acc_same"):
                key = f"{comp}_{overlap_group}_{acc_group}"
                summary[key] = int(
                    sum(
                        1
                        for r in detail_rows
                        if r.get(f"group_{comp}") == overlap_group and r.get(f"acc_group_{comp}") == acc_group
                    )
                )
    return summary


def _plot_overlap_accuracy_effect_map(all_fold_summaries: List[Dict[str, object]], output_path: Path) -> None:
    colors = {"reward": "#2E7D32", "log": "#F57C00", "exp": "#D32F2F"}
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for comp in ("reward", "log", "exp"):
        xs, ys = [], []
        for row in all_fold_summaries:
            n_face = max(1, int(row.get("n_face_ok", 0)))
            mean_delta = float(row.get(f"{comp}_mean_delta", np.nan))
            improved = int(row.get(f"{comp}_acc_improved", 0))
            worsened = int(row.get(f"{comp}_acc_worsened", 0))
            acc_delta = (improved - worsened) / float(n_face)
            if np.isfinite(mean_delta):
                xs.append(mean_delta)
                ys.append(acc_delta)
                ax.scatter(mean_delta, acc_delta, color=colors[comp], alpha=0.75, s=45)

        if xs:
            ax.scatter(np.mean(xs), np.mean(ys), color=colors[comp], edgecolors="black", linewidth=1.0, s=130, label=f"{comp} (fold-mean)")

    ax.axhline(0.0, color="#9E9E9E", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#9E9E9E", linestyle="--", linewidth=1.0)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Mean overlap delta vs baseline")
    ax.set_ylabel("Accuracy delta rate (improved - worsened)")
    ax.set_title("Overlap vs Accuracy Effect Map")
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_overlap_acc_group_stacked(all_fold_summaries: List[Dict[str, object]], output_path: Path) -> None:
    comps = ("reward", "log", "exp")
    overlap_groups = ("increased", "decreased")
    acc_groups = ("acc_improved", "acc_worsened", "acc_same")
    acc_colors = {"acc_improved": "#2E7D32", "acc_worsened": "#C62828", "acc_same": "#9E9E9E"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for i, comp in enumerate(comps):
        ax = axes[i]
        x = np.arange(len(overlap_groups))
        bottoms = np.zeros(len(overlap_groups), dtype=np.float32)

        # fold-average normalized composition for each overlap group
        vals_by_acc = {ag: [] for ag in acc_groups}
        for og in overlap_groups:
            denom = np.mean([
                max(
                    1,
                    int(r.get(f"{comp}_{og}_acc_improved", 0))
                    + int(r.get(f"{comp}_{og}_acc_worsened", 0))
                    + int(r.get(f"{comp}_{og}_acc_same", 0)),
                )
                for r in all_fold_summaries
            ])
            for ag in acc_groups:
                num = np.mean([int(r.get(f"{comp}_{og}_{ag}", 0)) for r in all_fold_summaries])
                vals_by_acc[ag].append(float(num / float(denom)))

        for ag in acc_groups:
            vals = np.array(vals_by_acc[ag], dtype=np.float32)
            ax.bar(x, vals, bottom=bottoms, color=acc_colors[ag], label=ag if i == 0 else None, width=0.58)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(overlap_groups)
        ax.set_title(comp)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fold-mean ratio")
    fig.suptitle("Accuracy Outcome Composition within Overlap Groups", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def run_fold_overlap_analysis(
    *,
    fold_id: int,
    val_image_paths: List[str],
    model_paths: Dict[str, str],
    output_dir: Path,
    img_size: Tuple[int, int],
) -> Dict[str, object]:
    print(f"\n========== FOLD {fold_id} ==========")
    print(f"Val images: {len(val_image_paths)} paths (fold JSON)")

    file_paths = val_image_paths
    if len(file_paths) == 0:
        raise ValueError(f"Fold {fold_id}: empty val image list")

    models: Dict[str, tf.keras.Model] = {}
    cams: Dict[str, CustomGradCAM] = {}
    for label, m_path in model_paths.items():
        if not os.path.isfile(m_path):
            raise FileNotFoundError(f"Fold {fold_id}: model not found for '{label}': {m_path}")
        model = tf.keras.models.load_model(m_path, compile=False)
        models[label] = model
        cams[label] = CustomGradCAM(model)

    detail_rows: List[Dict[str, object]] = []
    skipped_bad_images = 0
    for idx, image_path in enumerate(file_paths):
        try:
            image_uint8 = _load_image(image_path, img_size)
        except (OSError, UnidentifiedImageError, ValueError) as e:
            skipped_bad_images += 1
            print(f"[WARN] Fold {fold_id}: skipping unreadable image: {image_path} ({e})")
            continue
        mask = create_landmark_mask(
            image_uint8,
            img_size,
            background_value=float(CONFIG.get("background_mask_value", 0.0)),
            landmark_box_half_size=int(CONFIG.get("roi_padding_px", 12)),
        )
        face_ok = int(mask is not None)
        true_label = int(labels_from_paths([image_path], _CLASS_NAMES)[0])

        ratios: Dict[str, Optional[float]] = {k: None for k in model_paths.keys()}
        preds: Dict[str, Optional[int]] = {k: None for k in model_paths.keys()}
        if face_ok == 1:
            for label in model_paths.keys():
                ratio, pred = _compute_focus_ratio_for_model(
                    model=models[label],
                    cam=cams[label],
                    image_uint8=image_uint8,
                    mask=mask,  # type: ignore[arg-type]
                    img_size=img_size,
                )
                ratios[label] = ratio
                preds[label] = pred

        d_reward = None if ratios["baseline"] is None or ratios["reward"] is None else ratios["reward"] - ratios["baseline"]
        d_log = None if ratios["baseline"] is None or ratios["log"] is None else ratios["log"] - ratios["baseline"]
        d_exp = None if ratios["baseline"] is None or ratios["exp"] is None else ratios["exp"] - ratios["baseline"]
        acc_group_reward = _compare_accuracy_group(preds["baseline"], preds["reward"], true_label)
        acc_group_log = _compare_accuracy_group(preds["baseline"], preds["log"], true_label)
        acc_group_exp = _compare_accuracy_group(preds["baseline"], preds["exp"], true_label)

        detail_rows.append(
            {
                "fold_id": int(fold_id),
                "image_path": image_path,
                "true_label": true_label,
                "face_ok": face_ok,
                "baseline": _safe_float(ratios["baseline"]),
                "reward": _safe_float(ratios["reward"]),
                "log": _safe_float(ratios["log"]),
                "exp": _safe_float(ratios["exp"]),
                "baseline_pred": preds["baseline"],
                "reward_pred": preds["reward"],
                "log_pred": preds["log"],
                "exp_pred": preds["exp"],
                "delta_reward": _safe_float(d_reward),
                "delta_log": _safe_float(d_log),
                "delta_exp": _safe_float(d_exp),
                "group_reward": _compare_delta(d_reward),
                "group_log": _compare_delta(d_log),
                "group_exp": _compare_delta(d_exp),
                "acc_group_reward": acc_group_reward,
                "acc_group_log": acc_group_log,
                "acc_group_exp": acc_group_exp,
            }
        )

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(file_paths)}")
    if skipped_bad_images > 0:
        print(f"[INFO] Fold {fold_id}: skipped unreadable images: {skipped_bad_images}")

    fold_dir = output_dir / f"fold_{fold_id}"
    comparison_summaries = []
    for comp in ("reward", "log", "exp"):
        comparison_summaries.append(
            _write_comparison_csv(
                detail_rows=detail_rows,
                fold_dir=fold_dir,
                comp=comp,
            )
        )

    gradcam_sample_summaries = []
    if bool(CONFIG.get("save_gradcam_samples", True)):
        for comp in ("reward", "log", "exp"):
            gradcam_sample_summaries.append(
                _save_group_gradcam_examples(
                    detail_rows=detail_rows,
                    fold_dir=fold_dir,
                    comp=comp,
                    models=models,
                    cams=cams,
                    img_size=img_size,
                )
            )

    detail_csv = fold_dir / "overlap_details.csv"
    if bool(CONFIG.get("save_fold_detail_csv", False)):
        _write_details_csv(detail_rows, detail_csv)

    fold_summary = _build_fold_summary(fold_id, detail_rows)
    fold_summary["comparisons"] = comparison_summaries
    fold_summary["gradcam_samples"] = gradcam_sample_summaries
    with open(fold_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2, ensure_ascii=False)

    if bool(CONFIG.get("save_fold_detail_csv", False)):
        print(f"Saved detail csv: {detail_csv}")
    print(f"Saved comparison csv files: baseline_vs_reward/log/exp.csv")
    if bool(CONFIG.get("save_gradcam_samples", True)):
        print("Saved GradCAM sample groups under: gradcam_samples/")
    print(f"Saved summary json: {fold_dir / 'summary.json'}")
    return fold_summary


def main() -> None:
    fold_start = int(CONFIG["fold_start"])
    fold_count = int(CONFIG["fold_count"])
    fold_ids = list(range(fold_start, fold_start + fold_count))

    out_tag = (CONFIG.get("output_tag") or "").strip() or "overlap"
    output_dir = project_root / "artifacts" / "overlap_accuracy_comparison" / out_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_summaries: List[Dict[str, object]] = []
    for fold_id in fold_ids:
        val_paths, model_paths = _resolve_fold_paths(fold_id)
        if len(val_paths) == 0:
            print(f"[SKIP] Fold {fold_id}: no val paths in fold JSON.")
            continue
        fold_dir = output_dir / f"fold_{fold_id}"
        summary_path = fold_dir / "summary.json"
        if bool(CONFIG.get("resume_skip_fold_if_summary_exists", False)) and summary_path.is_file():
            print(f"[RESUME] Fold {fold_id} summary exists, skipping: {summary_path}")
            with open(summary_path, "r", encoding="utf-8") as f:
                all_fold_summaries.append(json.load(f))
            continue
        fold_summary = run_fold_overlap_analysis(
            fold_id=fold_id,
            val_image_paths=val_paths,
            model_paths=model_paths,
            output_dir=output_dir,
            img_size=tuple(CONFIG["img_size"]),
        )
        all_fold_summaries.append(fold_summary)

    if len(all_fold_summaries) == 0:
        print("\nNo fold summary generated.")
        return

    aggregate_csv = output_dir / "fold_summaries.csv"
    with open(aggregate_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold_id",
                "n_total",
                "n_face_ok",
                "reward_increased",
                "reward_decreased",
                "reward_equal",
                "reward_mean_delta",
                "reward_acc_improved",
                "reward_acc_worsened",
                "reward_acc_same",
                "log_increased",
                "log_decreased",
                "log_equal",
                "log_mean_delta",
                "log_acc_improved",
                "log_acc_worsened",
                "log_acc_same",
                "exp_increased",
                "exp_decreased",
                "exp_equal",
                "exp_mean_delta",
                "exp_acc_improved",
                "exp_acc_worsened",
                "exp_acc_same",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in all_fold_summaries:
            writer.writerow(row)

    with open(output_dir / "fold_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_fold_summaries, f, indent=2, ensure_ascii=False)

    plots_dir = output_dir / "plots"
    _plot_overlap_accuracy_effect_map(
        all_fold_summaries=all_fold_summaries,
        output_path=plots_dir / "overlap_accuracy_effect_map.png",
    )
    _plot_overlap_acc_group_stacked(
        all_fold_summaries=all_fold_summaries,
        output_path=plots_dir / "overlap_acc_group_stacked.png",
    )

    print("\nCompleted overlap comparison.")
    print(f"Output folder: {output_dir}")
    print(f"Summary csv: {aggregate_csv}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
