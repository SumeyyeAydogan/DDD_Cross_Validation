def plot_threshold_results(ttc, save_path=None):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    results = ttc.cv_results_

    thresholds = np.asarray(results["thresholds"])
    scores = np.asarray(results["scores"])

    best_threshold = ttc.best_threshold_
    best_score = ttc.best_score_

    plt.figure(figsize=(8, 5))

    plt.plot(
        thresholds,
        scores,
        label="Balanced Accuracy"
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best Threshold = {best_threshold:.3f}"
    )

    plt.scatter(
        [best_threshold],
        [best_score],
        zorder=3
    )

    # Default threshold'u da görmek faydalı
    plt.axvline(
        0.5,
        linestyle=":",
        label="Default Threshold = 0.5"
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Balanced Accuracy")
    plt.title("Threshold Tuning on Validation Set")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Threshold tuning plot saved to: {save_path}")
    else:
        plt.show()