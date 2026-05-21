"""
Generate eye+mouth–focused versions of images using facial landmarks (no OpenCV).

Uses MediaPipe FaceMesh (via ``src.mask_helpers``), PIL + NumPy.

Standalone ``ddd_cv`` layout (defaults):
  - ``dataset/NotDrowsy/*``, ``dataset/Drowsy/*``  (flat class folders)
  - Output: ``dataset_landmark_roi/`` mirroring relative paths

Optional split layout (like ``train/NotDrowsy``):
  - Pass ``--layout split`` and set ``--source`` to the split root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.mask_helpers import create_landmark_mask, image_to_uint8_rgb, img_size_from_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def apply_mask_to_image(img: Image.Image, mask: np.ndarray, background_mask_value: float) -> Image.Image:
    img_np = np.array(img).astype(np.float32) / 255.0
    if mask.ndim == 2:
        mask_3 = np.stack([mask] * 3, axis=-1)
    else:
        mask_3 = mask
    bg = float(background_mask_value)
    masked = img_np * mask_3 + (1.0 - mask_3) * bg
    out_np = (np.clip(masked, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(out_np)


def process_one_image(
    src_path: Path,
    dst_path: Path,
    *,
    landmark_box_half_size: int,
    background_mask_value: float,
) -> None:
    try:
        img = Image.open(src_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"[WARN] Skipping unreadable image: {src_path} ({e})")
        return

    image_rgb_uint8 = image_to_uint8_rgb(np.array(img))
    img_size = img_size_from_rgb(image_rgb_uint8)

    mask = create_landmark_mask(
        image_rgb_uint8,
        img_size,
        landmark_box_half_size=landmark_box_half_size,
        background_mask_value=background_mask_value,
    )
    if mask is None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        print(f"[WARN] No face detected, copied original: {src_path}")
        return

    masked_img = apply_mask_to_image(img, mask, background_mask_value)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    masked_img.save(dst_path)
    print(f"[OK] Processed: {src_path} -> {dst_path}")


def run_flat(
    source_root: Path,
    target_root: Path,
    classes: Sequence[str],
    *,
    landmark_box_half_size: int,
    background_mask_value: float,
) -> None:
    for cls in classes:
        src_dir = source_root / cls
        if not src_dir.is_dir():
            print(f"[WARN] Missing class folder: {src_dir}")
            continue
        for p in src_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = Path(cls) / p.relative_to(src_dir)
            dst_path = target_root / rel
            process_one_image(
                p,
                dst_path,
                landmark_box_half_size=landmark_box_half_size,
                background_mask_value=background_mask_value,
            )


def run_split(
    source_root: Path,
    target_root: Path,
    splits: Sequence[str],
    classes: Sequence[str],
    *,
    landmark_box_half_size: int,
    background_mask_value: float,
) -> None:
    for split in splits:
        for cls in classes:
            src_dir = source_root / split / cls
            if not src_dir.is_dir():
                print(f"[WARN] Missing: {src_dir}")
                continue
            for p in src_dir.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                rel = Path(split) / Path(cls) / p.relative_to(src_dir)
                dst_path = target_root / rel
                process_one_image(
                    p,
                    dst_path,
                    landmark_box_half_size=landmark_box_half_size,
                    background_mask_value=background_mask_value,
                )


def detect_layout(source_root: Path, classes: Sequence[str]) -> str:
    if (source_root / "train").is_dir():
        return "split"
    if any((source_root / c).is_dir() for c in classes):
        return "flat"
    raise FileNotFoundError(
        f"Could not detect layout under {source_root}: "
        f"expected either train/… or class folders {list(classes)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Landmark ROI mask over ddd_cv dataset (standalone).")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Dataset root (default: <ddd_cv>/dataset)",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Output root (default: <ddd_cv>/dataset_landmark_roi)",
    )
    p.add_argument(
        "--layout",
        choices=("auto", "flat", "split"),
        default="auto",
        help="flat: dataset/Class/… ; split: dataset/train/Class/…",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma splits for --layout split (ignored for flat).",
    )
    p.add_argument("--landmark_box_half_size", type=int, default=12)
    p.add_argument("--background_mask_value", type=float, default=0.0)
    p.add_argument(
        "--classes",
        type=str,
        default="NotDrowsy,Drowsy",
        help="Comma-separated class directory names.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = _PROJECT_ROOT
    source = Path(args.source) if args.source else root / "dataset"
    target = Path(args.target) if args.target else root / "dataset_landmark_roi"
    classes: List[str] = [x.strip() for x in args.classes.split(",") if x.strip()]
    if len(classes) != 2:
        raise SystemExit("--classes must list exactly two names, e.g. NotDrowsy,Drowsy")

    layout = args.layout
    if layout == "auto":
        layout = detect_layout(source, classes)

    print(f"[INFO] project_root: {root}")
    print(f"[INFO] layout: {layout}")
    print(f"[INFO] source: {source}")
    print(f"[INFO] target: {target}")

    if layout == "flat":
        run_flat(
            source,
            target,
            classes,
            landmark_box_half_size=args.landmark_box_half_size,
            background_mask_value=args.background_mask_value,
        )
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        run_split(
            source,
            target,
            splits,
            classes,
            landmark_box_half_size=args.landmark_box_half_size,
            background_mask_value=args.background_mask_value,
        )

    print("\n[DONE] Landmark-masked dataset written under:")
    print(f"       {target}")


if __name__ == "__main__":
    main()
