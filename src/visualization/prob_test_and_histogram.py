import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src.evaluation.threshold import collect_proba


def plot_prob_histogram(
    y,
    scores,
    threshold,
    plot_path=None,
    stats_path=None,
):
    y = np.asarray(y).ravel()
    scores = np.asarray(scores).ravel()

    lines = []

    for cls in [0, 1]:
        s = scores[y == cls]

        lines.append(f"\nClass {cls}")
        lines.append(f"count : {len(s)}")
        lines.append(f"min   : {np.min(s):.6f}")
        lines.append(f"p1    : {np.percentile(s, 1):.6f}")
        lines.append(f"p5    : {np.percentile(s, 5):.6f}")
        lines.append(f"p25   : {np.percentile(s, 25):.6f}")
        lines.append(f"median: {np.median(s):.6f}")
        lines.append(f"p75   : {np.percentile(s, 75):.6f}")
        lines.append(f"p95   : {np.percentile(s, 95):.6f}")
        lines.append(f"p99   : {np.percentile(s, 99):.6f}")
        lines.append(f"max   : {np.max(s):.6f}")

    text = "\n".join(lines)

    # Terminale de yaz
    print(text)

    # TXT kaydet
    if stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)

        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Probability statistics saved to: {stats_path}")

    # Histogram
    plt.figure(figsize=(8, 5))

    plt.hist(
        scores[y == 0],
        bins=50,
        alpha=0.5,
        label="NotDrowsy",
    )

    plt.hist(
        scores[y == 1],
        bins=50,
        alpha=0.5,
        label="Drowsy",
    )

    plt.axvline(
        0.5,
        linestyle=":",
        label="Default = 0.5",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Tuned = {threshold:.3f}",
    )

    plt.xlabel("Predicted Probability for Drowsy")
    plt.ylabel("Count")
    plt.title("Validation Probability Distribution")
    plt.legend()
    plt.tight_layout()

    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        plt.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Probability histogram saved to: {plot_path}")
    else:
        plt.show()


def compare_tuned_and_default_thresholds(
    y_test,
    test_scores,
    threshold,
    save_path=None,
):
    y_test = np.asarray(y_test).ravel()
    test_scores = np.asarray(test_scores).ravel()

    pred_05 = (test_scores >= 0.5).astype(int)
    pred_tuned = (test_scores >= threshold).astype(int)

    lines = []

    for name, pred in [
        ("Default 0.5", pred_05),
        (f"Tuned {threshold:.4f}", pred_tuned),
    ]:
        acc = accuracy_score(y_test, pred)
        bal_acc = balanced_accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)

        report = classification_report(
            y_test,
            pred,
            target_names=[
                "NotDrowsy",
                "Drowsy",
            ],
            digits=4,
        )

        lines.append("=" * 60)
        lines.append(name)
        lines.append("=" * 60)

        lines.append(f"Accuracy: {acc:.6f}")
        lines.append(f"Balanced Accuracy: {bal_acc:.6f}")

        lines.append("\nConfusion Matrix:")
        lines.append(str(cm))

        lines.append("\nClassification Report:")
        lines.append(report)

    text = "\n".join(lines)

    print(text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Threshold comparison saved to: {save_path}")


def evaluate_calibration(
    y,
    scores,
    n_bins=10,
    plot_path=None,
    metrics_path=None,
):
    y = np.asarray(y).ravel()
    scores = np.asarray(scores).ravel()

    brier = brier_score_loss(y, scores)

    prob_true, prob_pred = calibration_curve(
        y,
        scores,
        n_bins=n_bins,
        strategy="quantile",
    )

    text = (
        f"Brier Score: {brier:.6f}\n"
        f"Number of bins: {n_bins}\n"
        f"Strategy: quantile\n"
    )

    print(text)

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(text)

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label="Model",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        plt.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Calibration curve saved to: {plot_path}")
    else:
        plt.show()

def evaluate_auc_metrics(y_val, y_test, val_scores, test_scores, save_path=None,):

    lines = []

    lines.append("\n--- AUC ---")
    lines.append(f"Validation N: {len(y_val)}")
    lines.append(f"Validation AUC: {roc_auc_score(y_val, val_scores)}")

    lines.append(f"Test N: {len(y_test)}")
    lines.append(f"Test AUC: {roc_auc_score(y_test, test_scores)}")

    lines.append("\n--- Validation class probability means ---")
    lines.append(f"Class 0 mean: {val_scores[y_val == 0].mean()}")
    lines.append(f"Class 1 mean: {val_scores[y_val == 1].mean()}")

    text = "\n".join(lines)

    print(text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Threshold comparison saved to: {save_path}")

def test_threshold(
    model,
    val_ds,
    test_ds,
    threshold,
    output_dir=None,
):
    y_val, val_scores = collect_proba(
        model,
        val_ds,
    )

    y_test, test_scores = collect_proba(
        model,
        test_ds,
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        prob_hist_path = os.path.join(
            output_dir,
            "validation_probability_histogram.png",
        )

        prob_stats_path = os.path.join(
            output_dir,
            "validation_probability_stats.txt",
        )

        calibration_plot_path = os.path.join(
            output_dir,
            "validation_calibration_curve.png",
        )

        calibration_metrics_path = os.path.join(
            output_dir,
            "validation_calibration_metrics.txt",
        )

        threshold_comparison_path = os.path.join(
            output_dir,
            "test_threshold_comparison.txt",
        )

        auc_metrics_path = os.path.join(
            output_dir,
            "auc_metrics.txt",
        )

    else:
        prob_hist_path = None
        prob_stats_path = None
        calibration_plot_path = None
        calibration_metrics_path = None
        threshold_comparison_path = None
        auc_metrics_path = None

    print("\n================================")
    print("VALIDATION PROBABILITY ANALYSIS")
    print("================================")

    plot_prob_histogram(
        y_val,
        val_scores,
        threshold,
        plot_path=prob_hist_path,
        stats_path=prob_stats_path,
    )

    print("\n================================")
    print("VALIDATION CALIBRATION ANALYSIS")
    print("================================")

    evaluate_calibration(
        y_val,
        val_scores,
        n_bins=10,
        plot_path=calibration_plot_path,
        metrics_path=calibration_metrics_path,
    )

    print("\n================================")
    print("TEST THRESHOLD COMPARISON")
    print("================================")

    compare_tuned_and_default_thresholds(
        y_test,
        test_scores,
        threshold,
        save_path=threshold_comparison_path,
    )

    evaluate_auc_metrics(
       y_val, y_test, val_scores, test_scores, save_path=auc_metrics_path, 
    )

