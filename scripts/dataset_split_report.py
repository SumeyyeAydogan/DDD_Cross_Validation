#!/usr/bin/env python3
"""Create a dependency-free HTML report for fold dataset distributions."""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


PERSON_RE = re.compile(r"^([A-Za-z]+)")
CLASS_NAMES = ("NotDrowsy", "Drowsy")


def person_prefix(path: str) -> str:
    stem = Path(path).stem
    match = PERSON_RE.match(stem)
    return match.group(1) if match else stem


def load_rows(fold_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for fold_json in sorted(fold_dir.glob("fold_*.json")):
        try:
            fold_id = int(fold_json.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        data = json.loads(fold_json.read_text(encoding="utf-8"))
        for split, payload in data.items():
            if split == "meta" or not isinstance(payload, dict):
                continue
            files = payload.get("files") or []
            labels = payload.get("labels") or []
            for file_path, label in zip(files, labels):
                prefix = person_prefix(str(file_path))
                label_int = int(float(label))
                rows.append(
                    {
                        "fold": fold_id,
                        "split": split,
                        "class": CLASS_NAMES[label_int],
                        "person": prefix.lower(),
                        "prefix": prefix,
                    }
                )
    return rows


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def split_class_counts(rows: List[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[(row["fold"], row["split"], row["class"])] += 1
    return counts


def stacked_bar_svg(counts: Counter) -> str:
    groups = sorted({(fold, split) for fold, split, _ in counts})
    if not groups:
        return ""
    max_total = max(sum(counts[(fold, split, cls)] for cls in CLASS_NAMES) for fold, split in groups)
    bar_w, gap, height = 34, 18, 260
    left, top, bottom = 50, 20, 66
    width = left + len(groups) * (bar_w + gap) + 20
    chart_h = height - top - bottom
    colors = {"NotDrowsy": "#4C78A8", "Drowsy": "#F58518"}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - 10}" y2="{top + chart_h}" stroke="#999"/>',
    ]
    for idx, (fold, split) in enumerate(groups):
        x = left + idx * (bar_w + gap)
        y_cursor = top + chart_h
        total = 0
        for cls in CLASS_NAMES:
            n = counts[(fold, split, cls)]
            total += n
            h = 0 if max_total == 0 else int(chart_h * n / max_total)
            y_cursor -= h
            parts.append(f'<rect x="{x}" y="{y_cursor}" width="{bar_w}" height="{h}" fill="{colors[cls]}"/>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{top + chart_h + 14}" text-anchor="middle" font-size="10">F{fold}</text>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{top + chart_h + 28}" text-anchor="middle" font-size="9">{esc(split)}</text>')
        parts.append(f'<text x="{x + bar_w / 2}" y="{max(12, y_cursor - 4)}" text-anchor="middle" font-size="9">{total}</text>')
    parts.append('<rect x="52" y="4" width="10" height="10" fill="#4C78A8"/><text x="66" y="13" font-size="11">NotDrowsy</text>')
    parts.append('<rect x="145" y="4" width="10" height="10" fill="#F58518"/><text x="159" y="13" font-size="11">Drowsy</text>')
    parts.append("</svg>")
    return "".join(parts)


def person_fold_summary(rows: List[dict]) -> List[dict]:
    by_key: Dict[tuple, dict] = defaultdict(lambda: {"splits": Counter(), "classes": Counter(), "total": 0})
    for row in rows:
        item = by_key[(row["fold"], row["person"])]
        item["splits"][row["split"]] += 1
        item["classes"][row["class"]] += 1
        item["total"] += 1
    out = []
    split_names = sorted({row["split"] for row in rows})
    for (fold, person), item in by_key.items():
        out.append(
            {
                "fold": fold,
                "person": person,
                "splits": ", ".join(sorted(item["splits"])),
                "classes": ", ".join(sorted(item["classes"])),
                "cross_split": len(item["splits"]) > 1,
                "both_classes": all(cls in item["classes"] for cls in CLASS_NAMES),
                "total": item["total"],
                **{f"{split}_n": item["splits"].get(split, 0) for split in split_names},
                **{f"{cls}_n": item["classes"].get(cls, 0) for cls in CLASS_NAMES},
            }
        )
    return sorted(out, key=lambda x: (not x["cross_split"], not x["both_classes"], -x["total"], x["fold"], x["person"]))


def top_people(rows: List[dict], n: int = 30) -> List[tuple]:
    counts: Counter = Counter()
    class_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["person"]] += 1
        class_counts[row["person"]][row["class"]] += 1
    people = [person for person, _ in counts.most_common(n)]
    return [
        (person, counts[person], class_counts[person]["NotDrowsy"], class_counts[person]["Drowsy"])
        for person in people
    ]


def render_section(fold_dir: Path, rows: List[dict]) -> str:
    people = sorted({row["person"] for row in rows})
    splits = sorted({row["split"] for row in rows})
    folds = sorted({row["fold"] for row in rows})
    both_class_people = sorted(
        person
        for person in people
        if {row["class"] for row in rows if row["person"] == person} == set(CLASS_NAMES)
    )
    overlap = person_fold_summary(rows)
    risky = [row for row in overlap if row["cross_split"] and row["both_classes"]]
    split_headers = sorted(k for k in overlap[0].keys() if k.endswith("_n")) if overlap else []
    overlap_rows = [
        [
            row["fold"],
            row["person"],
            row["splits"],
            row["classes"],
            row["cross_split"],
            row["both_classes"],
            row["total"],
            *[row.get(h, 0) for h in split_headers],
        ]
        for row in risky[:250]
    ]
    counts = split_class_counts(rows)
    top_rows = top_people(rows)
    return f"""
<section>
  <h2>{esc(fold_dir.name)}</h2>
  <div class="metrics">
    <div><b>{len(rows):,}</b><span>images</span></div>
    <div><b>{len(folds)}</b><span>folds</span></div>
    <div><b>{len(splits)}</b><span>splits</span></div>
    <div><b>{len(people)}</b><span>normalized people</span></div>
    <div><b>{len(both_class_people)}</b><span>people with both classes</span></div>
    <div><b>{len(risky)}</b><span>person-fold cross split + both classes</span></div>
  </div>
  <h3>Images by fold, split, and class</h3>
  {stacked_bar_svg(counts)}
  <h3>Top normalized people</h3>
  {table(["person", "total", "NotDrowsy", "Drowsy"], top_rows)}
  <h3>Risky person-fold rows</h3>
  {table(["fold", "person", "splits", "classes", "cross_split", "both_classes", "total", *split_headers], overlap_rows)}
</section>
"""


def render_report(root: Path, fold_dirs: List[Path]) -> str:
    sections = []
    for fold_dir in fold_dirs:
        rows = load_rows(fold_dir)
        if rows:
            sections.append(render_section(fold_dir, rows))
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DDD fold dataset distribution report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #202124; }}
section {{ margin: 0 0 42px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0 22px; }}
.metrics div {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px 12px; background: #fafafa; }}
.metrics b {{ display: block; font-size: 22px; }}
.metrics span {{ color: #666; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 22px; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f2f2f2; position: sticky; top: 0; }}
h1, h2, h3 {{ margin-bottom: 8px; }}
.note {{ color: #555; max-width: 880px; }}
</style>
</head>
<body>
<h1>DDD fold dataset distribution report</h1>
<p class="note">Person id is normalized by lowercasing the filename prefix, so A/a and ZA/za are treated as the same person. Risky rows indicate a normalized person appears in multiple splits and has both Drowsy and NotDrowsy variants inside the same fold.</p>
{''.join(sections)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--fold-dir", action="append", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/dataset_split_report.html"))
    args = parser.parse_args()

    root = args.root.resolve()
    fold_dirs = args.fold_dir or [root / "fold_datasets", root / "fold_datasets_v2"]
    fold_dirs = [p if p.is_absolute() else root / p for p in fold_dirs]
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(root, fold_dirs), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
