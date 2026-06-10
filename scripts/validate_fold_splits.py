# scripts/validate_fold_splits.py
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.fold_split import group_key

FOLD_DIR = project_root / "fold_datasets_v2"
CLASS_NAMES = ("NotDrowsy", "Drowsy")
K = 5


def _count_by_group(files: list[str]) -> Counter:
    return Counter(group_key(p, CLASS_NAMES) for p in files)


def _print_group_counts(title: str, counts: Counter) -> None:
    print(f"\n{title} - unique groups (class, person): {len(counts)}")
    for (class_name, person), n in sorted(counts.items()):
        print(f"  {class_name:10s}  person={person:6s}  images={n}")


def main() -> None:
    # Cross-fold test counts per (class, person)
    test_by_group_all_folds: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * K)

    for fold_n in range(1, K + 1):
        with open(FOLD_DIR / f"fold_{fold_n}.json", encoding="utf-8") as f:
            d = json.load(f)

        print(f"\n{'=' * 60}")
        print(f"Fold {fold_n}")
        print(f"{'=' * 60}")

        for split in ("train_fit", "val_monitor", "test"):
            files = d[split]["files"]
            labels = d[split]["labels"]
            label_counts = Counter(labels)
            print(
                f"\n{split:12s}  total={len(files)}  "
                f"NotDrowsy={label_counts.get(0.0, 0)}  Drowsy={label_counts.get(1.0, 0)}"
            )

            group_counts = _count_by_group(files)
            _print_group_counts(f"{split} by (class, person)", group_counts)

            if split == "test":
                for key, n in group_counts.items():
                    test_by_group_all_folds[key][fold_n - 1] = n

        # Same image must not appear in two splits
        sets = {s: set(d[s]["files"]) for s in ("train_fit", "val_monitor", "test")}
        assert sets["train_fit"].isdisjoint(sets["test"])
        assert sets["val_monitor"].isdisjoint(sets["test"])
        assert sets["train_fit"].isdisjoint(sets["val_monitor"])
        print("\nOK: no overlap between splits within this fold")

    # OOF: each image appears in test exactly once across all folds
    all_test: list[str] = []
    for fold_n in range(1, K + 1):
        with open(FOLD_DIR / f"fold_{fold_n}.json", encoding="utf-8") as f:
            all_test.extend(json.load(f)["test"]["files"])
    print(f"\n{'=' * 60}")
    print("OOF check")
    print(f"{'=' * 60}")
    print(f"total test entries across folds: {len(all_test)}")
    print(f"unique images in test:           {len(set(all_test))}")

    # Round-robin summary: test images per (class, person) in each fold
    print(f"\n{'=' * 60}")
    print("Test images per (class, person) across all folds")
    print(f"{'=' * 60}")
    header = f"{'class':10s}  {'person':6s}  " + "  ".join(f"F{i + 1:>4d}" for i in range(K)) + "  total"
    print(header)
    print("-" * len(header))
    for key in sorted(test_by_group_all_folds.keys()):
        class_name, person = key
        per_fold = test_by_group_all_folds[key]
        total = sum(per_fold)
        fold_cols = "  ".join(f"{n:4d}" for n in per_fold)
        print(f"{class_name:10s}  {person:6s}  {fold_cols}  {total:5d}")


if __name__ == "__main__":
    main()
