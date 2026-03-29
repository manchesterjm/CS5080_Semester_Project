"""Cross-checkpoint analysis and hypothesis testing.

Loads metrics from all checkpoints, extracts game performance from metrics.jsonl,
and tests the three hypotheses:
- H1: Imagination accuracy improves as training progresses
- H2: Better imagination correlates with better game performance
- H3: Semantic accuracy matters more than pixel-level accuracy

Usage:
    python analyze.py [--analysis-dir analysis_output] \
        [--metrics-jsonl training_output/snake_1M/metrics.jsonl]
"""

import argparse
import json
import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from scipy import stats  # pylint: disable=wrong-import-position

from constants import (  # pylint: disable=wrong-import-position
    METRIC_NAMES, PIXEL_METRICS, SEMANTIC_METRICS,
    FIGURE_DPI, P_VALUE_THRESHOLD, CHECKPOINT_STEP_TOLERANCE,
)
from shared import NumpyEncoder  # pylint: disable=wrong-import-position


def load_all_checkpoint_metrics(analysis_dir):
    """Load metrics.json from all checkpoint directories.

    Returns list of (training_step, metrics_dict) sorted by step.
    """
    analysis_dir = pathlib.Path(analysis_dir)
    results = []

    for metrics_path in sorted(analysis_dir.glob("checkpoint_*/metrics.json")):
        # Extract step number from directory name
        dir_name = metrics_path.parent.name  # checkpoint_0555000
        match = re.search(r"checkpoint_(\d+)", dir_name)
        if not match:
            continue
        step = int(match.group(1))

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        results.append((step, metrics))

    results.sort(key=lambda x: x[0])
    return results


def load_eval_performance(metrics_jsonl_path):
    """Extract eval_return and eval_length from metrics.jsonl.

    Returns dict mapping step → {eval_return, eval_length}.
    """
    perf = {}
    with open(metrics_jsonl_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "eval_return" in entry:
                step = int(entry["step"])
                perf[step] = {
                    "eval_return": entry["eval_return"],
                    "eval_length": entry.get("eval_length", None),
                }
    return perf


def match_performance_to_checkpoints(checkpoint_metrics, eval_perf):
    """For each checkpoint, find the closest eval performance entry.

    Returns list of (step, imag_metrics, eval_return, eval_length).
    """
    eval_steps = sorted(eval_perf.keys())
    matched = []

    for ckpt_step, metrics in checkpoint_metrics:
        # Find closest eval step
        closest = min(eval_steps, key=lambda s, cs=ckpt_step: abs(s - cs))
        if abs(closest - ckpt_step) > CHECKPOINT_STEP_TOLERANCE:
            print(f"  WARNING: No close eval for checkpoint {ckpt_step} "
                  f"(closest: {closest})")
            continue

        matched.append((
            ckpt_step,
            metrics,
            eval_perf[closest]["eval_return"],
            eval_perf[closest].get("eval_length"),
        ))

    return matched


BETTER_DIRECTION = {
    "mse": "lower", "ssim": "higher", "head_error": "lower",
    "body_accuracy": "higher", "food_correct": "higher",
}


def _is_improving(direction, rho):
    """Return True if the correlation direction indicates improvement."""
    return (direction == "lower" and rho < 0) or (direction == "higher" and rho > 0)


def _compute_h1_trends(matched_data):
    """Compute Spearman correlations between training step and each metric."""
    steps = np.array([m[0] for m in matched_data])
    results = {}
    for name in METRIC_NAMES:
        values = np.array([m[1]["overall"][name]["mean"] for m in matched_data])
        rho, p_value = stats.spearmanr(steps, values)
        improving = _is_improving(BETTER_DIRECTION[name], rho)
        results[name] = {
            "spearman_rho": float(rho),
            "p_value": float(p_value),
            "significant": p_value < P_VALUE_THRESHOLD,
            "improving": improving,
        }
        sig = "*" if p_value < P_VALUE_THRESHOLD else " "
        status = "improving" if improving else "NOT improving"
        print(f"  {name:20s}: rho={rho:+.3f}, p={p_value:.4f} {sig} ({status})")
    return results


def _plot_h1_trends(matched_data, results, figures_dir):
    """Plot metric trends vs training step for H1."""
    steps = np.array([m[0] for m in matched_data])
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, name in enumerate(METRIC_NAMES):
        values = np.array([m[1]["overall"][name]["mean"] for m in matched_data])
        rho = results[name]["spearman_rho"]
        p_val = results[name]["p_value"]
        ax = axes[idx]
        ax.plot(steps / 1000, values, "o-", markersize=4)
        ax.set_xlabel("Training Step (K)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} (ρ={rho:+.3f}, p={p_val:.3f})")
        ax.grid(True, alpha=0.3)
    axes[5].set_visible(False)
    plt.suptitle("H1: Imagination Quality vs Training Progress", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "h1_imagination_vs_training.png",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def _compute_early_late_ttest(matched_data):
    """Compare first k vs last k checkpoints with Welch's t-test."""
    k = min(5, len(matched_data) // 2)
    if k < 3:
        return None
    print(f"\n  Early vs Late t-test (first {k} vs last {k} checkpoints):")
    early_late = {}
    for name in METRIC_NAMES:
        early = np.array([m[1]["overall"][name]["mean"] for m in matched_data[:k]])
        late = np.array([m[1]["overall"][name]["mean"] for m in matched_data[-k:]])
        t_stat, t_p = stats.ttest_ind(early, late, equal_var=False)
        e_mean, l_mean = float(np.mean(early)), float(np.mean(late))
        direction = BETTER_DIRECTION[name]
        improved = l_mean < e_mean if direction == "lower" else l_mean > e_mean
        early_late[name] = {
            "early_mean": e_mean, "late_mean": l_mean,
            "t_statistic": float(t_stat), "p_value": float(t_p),
            "significant": t_p < P_VALUE_THRESHOLD, "improved": improved,
        }
        sig = "*" if t_p < P_VALUE_THRESHOLD else " "
        status = "improved" if improved else "NOT improved"
        print(f"    {name:20s}: early={e_mean:.4f}, late={l_mean:.4f}, "
              f"t={t_stat:+.3f}, p={t_p:.4f} {sig} ({status})")
    return {
        "k": k,
        "early_steps": [m[0] for m in matched_data[:k]],
        "late_steps": [m[0] for m in matched_data[-k:]],
        "metrics": early_late,
    }


def _plot_horizon_degradation(matched_data, figures_dir):
    """Plot per-step metric degradation over imagination horizon."""
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    n = len(matched_data)
    indices = sorted(set(min(i, n - 1) for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]))
    for idx, name in enumerate(METRIC_NAMES):
        ax = axes[idx]
        for si in indices:
            step, metrics = matched_data[si][0], matched_data[si][1]
            per_step = metrics["per_step"]
            ax.plot([s["step"] for s in per_step],
                    [s[f"{name}_mean"] for s in per_step],
                    "-", label=f"{step // 1000}K", alpha=0.7)
        ax.set_xlabel("Imagination Step")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs Horizon")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[5].set_visible(False)
    plt.suptitle("Imagination Quality Degradation Over Horizon", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "h1_horizon_degradation.png",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def test_h1(matched_data, figures_dir):
    """H1: Imagination accuracy improves as training progresses."""
    print("\n" + "=" * 60)
    print("H1: Does imagination accuracy improve over training?")
    print("=" * 60)
    results = _compute_h1_trends(matched_data)
    _plot_h1_trends(matched_data, results, figures_dir)
    early_late = _compute_early_late_ttest(matched_data)
    if early_late:
        results["early_vs_late"] = early_late
    _plot_horizon_degradation(matched_data, figures_dir)
    return results


def _compute_h2_correlations(matched_data):
    """Compute Spearman and Pearson correlations with eval return."""
    eval_returns = np.array([m[2] for m in matched_data])
    results = {}
    for name in METRIC_NAMES:
        values = np.array([m[1]["overall"][name]["mean"] for m in matched_data])
        s_rho, s_p = stats.spearmanr(values, eval_returns)
        p_r, p_p = stats.pearsonr(values, eval_returns)
        results[name] = {
            "spearman_rho": float(s_rho), "spearman_p": float(s_p),
            "pearson_r": float(p_r), "pearson_p": float(p_p),
        }
        print(f"  {name:20s}: Spearman ρ={s_rho:+.3f} (p={s_p:.4f}), "
              f"Pearson r={p_r:+.3f} (p={p_p:.4f})")
    return results


def _plot_h2_scatter(matched_data, results, figures_dir):
    """Plot scatter plots of each metric vs eval return."""
    eval_returns = np.array([m[2] for m in matched_data])
    _, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, name in enumerate(METRIC_NAMES):
        values = np.array([m[1]["overall"][name]["mean"] for m in matched_data])
        rho = results[name]["spearman_rho"]
        ax = axes[idx]
        ax.scatter(values, eval_returns, s=30)
        ax.set_xlabel(name)
        ax.set_ylabel("Eval Return")
        ax.set_title(f"{name} vs Return (ρ={rho:+.3f})")
        ax.grid(True, alpha=0.3)
        if len(values) > 2:
            coeffs = np.polyfit(values, eval_returns, 1)
            trend = np.poly1d(coeffs)
            x_sorted = np.sort(values)
            ax.plot(x_sorted, trend(x_sorted), "r--", alpha=0.5)
    axes[5].set_visible(False)
    plt.suptitle("H2: Imagination Quality vs Game Performance", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / "h2_imagination_vs_performance.png",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def test_h2(matched_data, figures_dir):
    """H2: Better imagination correlates with better game performance."""
    print("\n" + "=" * 60)
    print("H2: Does imagination accuracy correlate with game performance?")
    print("=" * 60)
    results = _compute_h2_correlations(matched_data)
    _plot_h2_scatter(matched_data, results, figures_dir)
    return results


def _compute_h3_correlations(matched_data):
    """Compute Spearman correlations for all pixel and semantic metrics."""
    eval_returns = np.array([m[2] for m in matched_data])
    all_metrics = list(PIXEL_METRICS) + list(SEMANTIC_METRICS)
    correlations = {}
    for name in all_metrics:
        values = np.array([m[1]["overall"][name]["mean"] for m in matched_data])
        rho, p = stats.spearmanr(values, eval_returns)
        correlations[name] = {"rho": rho, "p": p, "abs_rho": abs(rho)}
    return correlations


def _print_h3_correlations(correlations):
    """Print pixel vs semantic correlation comparison and averages."""
    print("\n  Pixel-level metrics:")
    for name in PIXEL_METRICS:
        c = correlations[name]
        print(f"    {name:20s}: |ρ| = {c['abs_rho']:.3f} (p={c['p']:.4f})")
    print("\n  Semantic metrics:")
    for name in SEMANTIC_METRICS:
        c = correlations[name]
        print(f"    {name:20s}: |ρ| = {c['abs_rho']:.3f} (p={c['p']:.4f})")
    avg_pixel = np.mean([correlations[m]["abs_rho"] for m in PIXEL_METRICS])
    avg_semantic = np.mean([correlations[m]["abs_rho"] for m in SEMANTIC_METRICS])
    print(f"\n  Average |ρ| pixel:    {avg_pixel:.3f}")
    print(f"  Average |ρ| semantic: {avg_semantic:.3f}")
    print(f"  Semantic advantage:   {avg_semantic - avg_pixel:+.3f}")
    return avg_pixel, avg_semantic


def _compute_fisher_z(correlations, n):
    """Run Fisher's z-test comparing best pixel vs best semantic metric."""
    if n <= 3:
        return None
    best_pixel = max(PIXEL_METRICS, key=lambda m: correlations[m]["abs_rho"])
    best_sem = max(SEMANTIC_METRICS, key=lambda m: correlations[m]["abs_rho"])
    r1, r2 = correlations[best_pixel]["abs_rho"], correlations[best_sem]["abs_rho"]
    se = np.sqrt(2.0 / (n - 3))
    z_stat = (np.arctanh(r2) - np.arctanh(r1)) / se
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"\n  Fisher's z-test ({best_sem} vs {best_pixel}):")
    print(f"    z = {z_stat:.3f}, p = {z_p:.4f} "
          f"{'(significant)' if z_p < P_VALUE_THRESHOLD else '(not significant)'}")
    return {
        "best_pixel": best_pixel, "best_pixel_rho": float(r1),
        "best_semantic": best_sem, "best_semantic_rho": float(r2),
        "z_statistic": float(z_stat), "p_value": float(z_p),
        "significant": z_p < P_VALUE_THRESHOLD,
    }


def _plot_h3_bar_chart(correlations, avg_pixel, avg_semantic, figures_dir):
    """Plot bar chart comparing pixel vs semantic correlation strengths."""
    all_metrics = list(PIXEL_METRICS) + list(SEMANTIC_METRICS)
    colors = ["#4ECDC4" if m in PIXEL_METRICS else "#FF6B6B" for m in all_metrics]
    abs_rhos = [correlations[m]["abs_rho"] for m in all_metrics]
    _, ax = plt.subplots(figsize=(10, 6))
    ax.bar(all_metrics, abs_rhos, color=colors, edgecolor="black", linewidth=0.5)
    for i, name in enumerate(all_metrics):
        if correlations[name]["p"] < P_VALUE_THRESHOLD:
            ax.text(i, abs_rhos[i] + 0.02, "*", ha="center", fontsize=14, fontweight="bold")
    ax.set_ylabel("|Spearman ρ| with Eval Return")
    ax.set_title("H3: Pixel vs Semantic Correlation with Performance")
    ax.axhline(y=avg_pixel, color="#4ECDC4", linestyle="--", alpha=0.5,
               label=f"Avg pixel ({avg_pixel:.3f})")
    ax.axhline(y=avg_semantic, color="#FF6B6B", linestyle="--", alpha=0.5,
               label=f"Avg semantic ({avg_semantic:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(abs_rhos) * 1.2 + 0.05)
    plt.tight_layout()
    plt.savefig(figures_dir / "h3_pixel_vs_semantic.png",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def test_h3(matched_data, figures_dir):
    """H3: Semantic metrics predict performance better than pixel metrics."""
    print("\n" + "=" * 60)
    print("H3: Do semantic metrics predict performance better than pixel metrics?")
    print("=" * 60)
    correlations = _compute_h3_correlations(matched_data)
    avg_pixel, avg_semantic = _print_h3_correlations(correlations)
    fisher_z = _compute_fisher_z(correlations, len(matched_data))
    _plot_h3_bar_chart(correlations, avg_pixel, avg_semantic, figures_dir)
    return {
        "correlations": {k: {"rho": v["rho"], "p": v["p"]}
                         for k, v in correlations.items()},
        "avg_pixel_abs_rho": float(avg_pixel),
        "avg_semantic_abs_rho": float(avg_semantic),
        "fisher_z": fisher_z,
    }


def generate_summary_table(matched_data, output_path):
    """Generate CSV summary table with all metrics per checkpoint."""
    header = ["step", "eval_return", "eval_length"] + \
             [f"{m}_mean" for m in METRIC_NAMES] + \
             [f"{m}_std" for m in METRIC_NAMES]

    rows = []
    for step, metrics, eval_ret, eval_len in matched_data:
        row = [step, eval_ret, eval_len or ""]
        for name in METRIC_NAMES:
            row.append(metrics["overall"][name]["mean"])
        for name in METRIC_NAMES:
            row.append(metrics["overall"][name]["std"])
        rows.append(row)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def run_analysis(analysis_dir, metrics_jsonl_path):
    """Run full cross-checkpoint analysis."""
    analysis_dir = pathlib.Path(analysis_dir)
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading checkpoint metrics...")
    checkpoint_metrics = load_all_checkpoint_metrics(analysis_dir)
    print(f"  Found {len(checkpoint_metrics)} checkpoints")

    if not checkpoint_metrics:
        print("ERROR: No checkpoint metrics found. Run imagine.py and compute_metrics.py first.")
        return None

    print("Loading eval performance from metrics.jsonl...")
    eval_perf = load_eval_performance(metrics_jsonl_path)
    print(f"  Found {len(eval_perf)} eval entries")

    matched_data = match_performance_to_checkpoints(checkpoint_metrics, eval_perf)
    print(f"  Matched {len(matched_data)} checkpoints to eval performance")

    if len(matched_data) < 3:
        print("ERROR: Need at least 3 matched checkpoints for correlation analysis.")
        return None

    # Generate summary CSV
    csv_path = analysis_dir / "summary_table.csv"
    generate_summary_table(matched_data, csv_path)
    print(f"\nSaved summary table to {csv_path}")

    # Run hypothesis tests
    h1_results = test_h1(matched_data, figures_dir)
    h2_results = test_h2(matched_data, figures_dir)
    h3_results = test_h3(matched_data, figures_dir)

    # Save combined results
    all_results = {
        "h1_imagination_improves": h1_results,
        "h2_correlates_with_performance": h2_results,
        "h3_semantic_vs_pixel": h3_results,
        "num_checkpoints": len(matched_data),
        "checkpoint_steps": [m[0] for m in matched_data],
    }

    results_path = analysis_dir / "hypothesis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nSaved hypothesis results to {results_path}")
    print(f"Figures saved to {figures_dir}/")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    h1_improving = sum(1 for v in h1_results.values() if v.get("improving"))
    h1_sig = sum(1 for v in h1_results.values() if v.get("significant"))
    print(f"H1: {h1_improving}/5 metrics improving, {h1_sig}/5 significant")

    h2_sig = sum(1 for v in h2_results.values()
                 if v.get("spearman_p", 1) < P_VALUE_THRESHOLD)
    print(f"H2: {h2_sig}/5 metrics significantly correlated with performance")

    if h3_results.get("fisher_z"):
        fz = h3_results["fisher_z"]
        print(f"H3: Semantic avg |ρ|={h3_results['avg_semantic_abs_rho']:.3f} vs "
              f"Pixel avg |ρ|={h3_results['avg_pixel_abs_rho']:.3f} "
              f"(Fisher z p={fz['p_value']:.4f})")

    return all_results


def main():
    """Parse arguments and run cross-checkpoint hypothesis testing."""
    parser = argparse.ArgumentParser(description="Cross-checkpoint hypothesis testing")
    parser.add_argument("--analysis-dir", type=str, default="analysis_output",
                        help="Directory containing checkpoint metric dirs")
    parser.add_argument("--metrics-jsonl", type=str,
                        default="training_output/snake_1M/metrics.jsonl",
                        help="Path to training metrics.jsonl")
    args = parser.parse_args()

    run_analysis(args.analysis_dir, args.metrics_jsonl)


if __name__ == "__main__":
    main()
