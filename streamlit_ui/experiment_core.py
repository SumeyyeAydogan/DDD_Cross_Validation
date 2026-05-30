"""Experiment catalog, presets, and subprocess launcher for DDD CV pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentSpec:
    id: str
    title: str
    description: str
    script_rel: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def experiments_dir(project_root: Path) -> Path:
    d = project_root / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def presets_path(project_root: Path) -> Path:
    return experiments_dir(project_root) / "presets.json"


def jobs_path(project_root: Path) -> Path:
    return experiments_dir(project_root) / "jobs.json"


def python_executable() -> str:
    return os.environ.get("DDD_PYTHON", sys.executable)


def catalog() -> Dict[str, ExperimentSpec]:
    return {
        "cv_train": ExperimentSpec(
            id="cv_train",
            title="CV training (5-fold)",
            description="Train all folds → runs/<run_name>/fold_k/… Uses fold_*_weights.json if weights dir set.",
            script_rel="scripts/run/cv_train.py",
            default_params={
                "run_name": "reward_percentile_dark",
                "weights_dir": "weights_percentile",
                "k": 5,
                "epochs": 30,
                "batch_size": 32,
                "seed": 42,
                "fold_start": 1,
                "fold_end": 5,
            },
            param_schema={
                "run_name": {"type": "text", "label": "Run name (runs/<name>)"},
                "weights_dir": {"type": "text", "label": "Weights dir (empty = no sample weights)"},
                "epochs": {"type": "int", "min": 1, "max": 200},
                "k": {"type": "int", "min": 1, "max": 20},
                "fold_start": {"type": "int", "min": 1, "max": 20},
                "fold_end": {"type": "int", "min": 1, "max": 20},
                "seed": {"type": "int", "min": 0, "max": 99999},
            },
        ),
        "compute_weights": ExperimentSpec(
            id="compute_weights",
            title="Compute GradCAM weights",
            description="GradCAM focus → fold_*_weights.json + log/exp variants.",
            script_rel="scripts/run/compute_weights.py",
            default_params={
                "base_run": "baseline",
                "output_dir": "weights_percentile",
                "weight_formula": "autoopt",
                "background_mask_value": 0.0,
                "k": 5,
                "fold_start": 1,
                "fold_end": 5,
                "read_only_reward_dir": "",
            },
            param_schema={
                "base_run": {"type": "text", "label": "Baseline run (for fold_k.h5)"},
                "output_dir": {"type": "text", "label": "Output weights directory"},
                "weight_formula": {
                    "type": "select",
                    "label": "Weight formula",
                    "options": ["autoopt", "density_gap"],
                },
                "background_mask_value": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
                "fold_start": {"type": "int", "min": 1, "max": 20},
                "fold_end": {"type": "int", "min": 1, "max": 20},
                "read_only_reward_dir": {
                    "type": "text",
                    "label": "Read-only reward dir (optional; skip GradCAM)",
                },
            },
        ),
        "compute_density_gap_weights": ExperimentSpec(
            id="compute_density_gap_weights",
            title="Compute Density-Gap weights",
            description="GradCAM density-gap weights → weights_new_formula/fold_*_weights.json + log/exp variants.",
            script_rel="scripts/run/compute_weights.py",
            default_params={
                "base_run": "baseline",
                "output_dir": "weights_new_formula",
                "weight_formula": "density_gap",
                "background_mask_value": 0.0,
                "k": 5,
                "fold_start": 1,
                "fold_end": 5,
                "read_only_reward_dir": "",
            },
            param_schema={
                "base_run": {"type": "text", "label": "Baseline run (for fold_k.h5)"},
                "output_dir": {"type": "text", "label": "Output weights directory"},
                "weight_formula": {
                    "type": "select",
                    "label": "Weight formula",
                    "options": ["density_gap", "autoopt"],
                },
                "background_mask_value": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
                "fold_start": {"type": "int", "min": 1, "max": 20},
                "fold_end": {"type": "int", "min": 1, "max": 20},
                "read_only_reward_dir": {
                    "type": "text",
                    "label": "Read-only reward dir (optional; skip GradCAM)",
                },
            },
        ),
        "overlap": ExperimentSpec(
            id="overlap",
            title="Overlap / focus comparison",
            description="Val-set focus ratios & decision groups → artifacts/overlap_accuracy_comparison/<tag>/",
            script_rel="scripts/run/overlap_comparison.py",
            default_params={
                "output_tag": "default",
                "fold_start": 1,
                "fold_count": 5,
                "background_mask_value": 0.0,
                "baseline_label": "baseline",
                "model_runs": "baseline=baseline,reward=reward,log=log,exp=exp",
                "no_resume": False,
                "focus_metric": "focus_ratio",
            },
            param_schema={
                "output_tag": {"type": "text", "label": "Output tag (subfolder name)"},
                "baseline_label": {"type": "text", "label": "Baseline label (must exist in model_runs)"},
                "model_runs": {"type": "text", "label": "label=run, comma-separated"},
                "fold_start": {"type": "int", "min": 1, "max": 20},
                "fold_count": {"type": "int", "min": 1, "max": 20},
                "background_mask_value": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
                "no_resume": {"type": "bool", "label": "Re-run all folds (ignore summary.json)"},
                "focus_metric": {
                    "type": "select",
                    "label": "Focus metric (per-image score)",
                    "options": [
                        "focus_ratio",
                        "density_gap",
                        "density_gap_shifted",
                        "inside_density",
                    ],
                },
            },
        ),
        "model_comparison": ExperimentSpec(
            id="model_comparison",
            title="Model comparison (focus integrals)",
            description="Per-fold focus histograms / tail metrics → artifacts/model_comparison/<tag>/",
            script_rel="scripts/run/model_comparison.py",
            default_params={
                "output_tag": "default",
                "fold_start": 1,
                "fold_count": 5,
                "baseline_label": "original",
                "model_runs": "original=baseline,reward=reward,log-reward=log,exp-reward=exp",
                "background_mask_value": 0.0,
                "experiment_id": "ddd_cv_folds",
                "focus_metric": "focus_ratio",
            },
            param_schema={
                "output_tag": {"type": "text", "label": "Output tag (subfolder name)"},
                "baseline_label": {"type": "text", "label": "Baseline label (must exist in model_runs)"},
                "model_runs": {"type": "text", "label": "label=run, comma-separated"},
                "fold_start": {"type": "int", "min": 1, "max": 20},
                "fold_count": {"type": "int", "min": 1, "max": 20},
                "experiment_id": {"type": "text", "label": "Experiment id (output naming)"},
                "background_mask_value": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
                "focus_metric": {
                    "type": "select",
                    "label": "Focus metric (per-image score)",
                    "options": [
                        "focus_ratio",
                        "density_gap",
                        "density_gap_shifted",
                        "inside_density",
                    ],
                },
            },
        ),
        "auto_optimize": ExperimentSpec(
            id="auto_optimize",
            title="Auto-optimize weights (single split)",
            description="Optimize GradCAM weight params on a data_dir; writes to artifacts/.",
            script_rel="scripts/auto_optimize_gradcam_weights.py",
            default_params={
                "model_path": "runs/baseline/fold_1/models/fold_1.h5",
                "data_dir": "dataset",
                "artifacts_dir": "artifacts/gradcam_opt",
                "background_mask_value": 0.0,
                "weight_mode": "reward",
            },
            param_schema={
                "model_path": {"type": "text"},
                "data_dir": {"type": "text"},
                "artifacts_dir": {"type": "text"},
                "background_mask_value": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
                "weight_mode": {"type": "select", "options": ["reward", "penalize"]},
            },
        ),
        "log_exp": ExperimentSpec(
            id="log_exp",
            title="Log / exp weight transform",
            description="Build log_weights.json and exp_weights.json from reward JSON.",
            script_rel="scripts/log_exp_script.py",
            default_params={
                "input": "weights_percentile/fold_1_weights.json",
                "log_out": "weights_percentile/fold_1_log_weights.json",
                "exp_out": "weights_percentile/fold_1_exp_weights.json",
            },
            param_schema={
                "input": {"type": "text", "label": "Input reward JSON"},
                "log_out": {"type": "text"},
                "exp_out": {"type": "text"},
            },
        ),
    }


def build_command(project_root: Path, spec_id: str, params: Dict[str, Any]) -> Tuple[List[str], Path]:
    specs = catalog()
    if spec_id not in specs:
        raise KeyError(f"Unknown experiment: {spec_id}")
    spec = specs[spec_id]
    script = (project_root / spec.script_rel).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    py = python_executable()
    cmd: List[str] = [py, str(script)]

    if spec_id == "cv_train":
        cmd += ["--run-name", str(params["run_name"])]
        if params.get("weights_dir"):
            cmd += ["--weights-dir", str(params["weights_dir"])]
        cmd += [
            "--k", str(int(params.get("k", 5))),
            "--epochs", str(int(params.get("epochs", 30))),
            "--seed", str(int(params.get("seed", 42))),
            "--fold-start", str(int(params.get("fold_start", 1))),
            "--fold-end", str(int(params.get("fold_end", 5))),
        ]
    elif spec_id in ("compute_weights", "compute_density_gap_weights"):
        cmd += [
            "--base-run", str(params["base_run"]),
            "--output-dir", str(params["output_dir"]),
            "--weight-formula", str(params.get("weight_formula", "autoopt")),
            "--background-mask-value", str(float(params["background_mask_value"])),
            "--fold-start", str(int(params.get("fold_start", 1))),
            "--fold-end", str(int(params.get("fold_end", 5))),
        ]
        if params.get("read_only_reward_dir"):
            cmd += ["--read-only-reward-dir", str(params["read_only_reward_dir"])]
    elif spec_id == "overlap":
        cmd += [
            "--output-tag", str(params["output_tag"]),
            "--fold-start", str(int(params["fold_start"])),
            "--fold-count", str(int(params["fold_count"])),
            "--background-mask-value", str(float(params["background_mask_value"])),
            "--baseline-label", str(params.get("baseline_label", "baseline")),
            "--model-runs", str(params["model_runs"]),
        ]
        if params.get("no_resume"):
            cmd += ["--no-resume"]
        cmd += ["--focus-metric", str(params.get("focus_metric", "focus_ratio"))]
    elif spec_id == "model_comparison":
        cmd += [
            "--output-tag", str(params.get("output_tag", "default")),
            "--fold-start", str(int(params["fold_start"])),
            "--fold-count", str(int(params["fold_count"])),
            "--baseline-label", str(params.get("baseline_label", "original")),
            "--model-runs", str(params["model_runs"]),
            "--experiment-id", str(params.get("experiment_id", "ddd_cv_folds")),
            "--background-mask-value", str(float(params["background_mask_value"])),
            "--focus-metric", str(params.get("focus_metric", "focus_ratio")),
        ]
    elif spec_id == "auto_optimize":
        cmd += [
            "--model_path", str(params["model_path"]),
            "--data_dir", str(params["data_dir"]),
            "--artifacts_dir", str(params["artifacts_dir"]),
            "--background_mask_value", str(float(params["background_mask_value"])),
            "--weight_mode", str(params["weight_mode"]),
        ]
    elif spec_id == "log_exp":
        cmd += [
            "--input", str(params["input"]),
            "--log_out", str(params["log_out"]),
            "--exp_out", str(params["exp_out"]),
        ]

    return cmd, script


def command_preview(cmd: List[str]) -> str:
    def quote(s: str) -> str:
        if " " in s or '"' in s:
            return '"' + s.replace('"', '\\"') + '"'
        return s

    return " ".join(quote(c) for c in cmd)


def load_presets(project_root: Path) -> Dict[str, Any]:
    path = presets_path(project_root)
    if not path.is_file():
        return {"version": 1, "presets": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_presets(project_root: Path, data: Dict[str, Any]) -> None:
    path = presets_path(project_root)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jobs(project_root: Path) -> List[Dict[str, Any]]:
    path = jobs_path(project_root)
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("jobs", []))


def save_jobs(project_root: Path, jobs: List[Dict[str, Any]]) -> None:
    path = jobs_path(project_root)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, indent=2, ensure_ascii=False)


def append_job(project_root: Path, job: Dict[str, Any]) -> None:
    jobs = load_jobs(project_root)
    jobs.insert(0, job)
    save_jobs(project_root, jobs[:50])


def start_job(
    project_root: Path,
    spec_id: str,
    params: Dict[str, Any],
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    cmd, script = build_command(project_root, spec_id, params)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log_dir = experiments_dir(project_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"

    job = {
        "id": job_id,
        "spec_id": spec_id,
        "params": params,
        "command": cmd,
        "command_preview": command_preview(cmd),
        "cwd": str(project_root.resolve()),
        "log_path": str(log_path),
        "status": "dry_run" if dry_run else "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": None,
        "returncode": None,
    }

    if dry_run:
        append_job(project_root, job)
        return job

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# {job['command_preview']}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    job["pid"] = proc.pid
    append_job(project_root, job)
    job["_proc"] = proc
    return job


def poll_job(job: Dict[str, Any]) -> Dict[str, Any]:
    proc = job.pop("_proc", None)
    if proc is not None:
        rc = proc.poll()
        if rc is not None:
            job["returncode"] = rc
            job["status"] = "success" if rc == 0 else "failed"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
    return job


def tail_log(log_path: str, n_lines: int = 80) -> str:
    p = Path(log_path)
    if not p.is_file():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])
