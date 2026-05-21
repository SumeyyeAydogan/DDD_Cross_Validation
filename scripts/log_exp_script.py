import json
import math
import os
import argparse

INPUT_PATH = "artifacts/reward-landmark-soft/optimized_gradcam_weights.json" #"artifacts/reward_soft-mask/optimized_gradcam_weights.json"
LOG_OUTPUT_PATH = "artifacts/reward-landmark-soft/log_weights.json"
EXP_OUTPUT_PATH = "artifacts/reward-landmark-soft/exp_weights.json"


def create_log_weights(input_path=INPUT_PATH, output_path=LOG_OUTPUT_PATH):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}")

    with open(input_path, "r") as f:
        weights = json.load(f)

    log_weights = {}

    for key, value in weights.items():
        safe_value = max(value, 1e-12)  # avoid log(0)
        log_weights[key] = math.log(safe_value)

    # 2) Find minimum log value
    min_log = min(log_weights.values())

    # 3) Shift by -min so all values are non-negative
    if min_log < 0:
        shift = -min_log
        for k in log_weights:
            log_weights[k] += shift    

    with open(output_path, "w") as f:
        json.dump(log_weights, f, indent=2)

    print(f"[OK] log weights saved to: {output_path}")


def create_exp_weights(input_path=INPUT_PATH, output_path=EXP_OUTPUT_PATH):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find {input_path}")

    with open(input_path, "r") as f:
        weights = json.load(f)

    exp_weights = {}

    for key, value in weights.items():
        # Clamp exponent input to avoid overflow
        safe_value = min(value, 50)  # exp(50) ≈ 5e21 — large enough cap
        exp_weights[key] = math.exp(safe_value)

    with open(output_path, "w") as f:
        json.dump(exp_weights, f, indent=2)

    print(f"[OK] exp weights saved to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create log/exp transformed weight JSONs.")
    parser.add_argument("--input", dest="input_path", default=INPUT_PATH, help="Path to optimized weights JSON")
    parser.add_argument("--log_out", dest="log_out", default=None, help="Output path for log weights JSON")
    parser.add_argument("--exp_out", dest="exp_out", default=None, help="Output path for exp weights JSON")
    parser.add_argument(
        "--out_dir",
        dest="out_dir",
        default=None,
        help="Output directory (writes log_weights.json and exp_weights.json). Ignored if --log_out/--exp_out are set.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.log_out:
        log_output = args.log_out
    elif args.out_dir:
        log_output = os.path.join(args.out_dir, "log_weights.json")
    else:
        log_output = LOG_OUTPUT_PATH

    if args.exp_out:
        exp_output = args.exp_out
    elif args.out_dir:
        exp_output = os.path.join(args.out_dir, "exp_weights.json")
    else:
        exp_output = EXP_OUTPUT_PATH

    os.makedirs(os.path.dirname(log_output), exist_ok=True)
    os.makedirs(os.path.dirname(exp_output), exist_ok=True)
    create_log_weights(input_path=args.input_path, output_path=log_output)
    create_exp_weights(input_path=args.input_path, output_path=exp_output)
