"""One-off: compare weight JSON directories."""
import json
from pathlib import Path

import numpy as np


def stats(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        w = json.load(f)
    v = np.array(list(w.values()), dtype=float)
    return {
        "n": len(v),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "p50": float(np.median(v)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("weights"))
    print("dirs:", [d.name for d in dirs])
    for d in dirs:
        print(f"\n=== {d.name} ===")
        for fold in range(1, 6):
            p = d / f"fold_{fold}_weights.json"
            if not p.is_file():
                continue
            s = stats(p)
            print(
                f"  fold_{fold}: n={s['n']} mean={s['mean']:.4f} std={s['std']:.4f} "
                f"min={s['min']:.4f} max={s['max']:.4f} median={s['p50']:.4f}"
            )


if __name__ == "__main__":
    main()
