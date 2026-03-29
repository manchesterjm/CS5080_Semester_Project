"""Compute frame-level and semantic metrics for imagination data.

For each checkpoint's imagination_data.npz, computes:
- Frame-level: MSE, SSIM
- Semantic: head_error, body_accuracy, food_correct (via StateExtractor)
- Aggregates per checkpoint and per imagination step

Usage:
    python compute_metrics.py --input analysis_output/checkpoint_0555000/imagination_data.npz
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snake_env.state_extractor import StateExtractor


def load_imagination_data(npz_path):
    """Load imagination data, returning list of episode dicts."""
    data = np.load(npz_path, allow_pickle=True)
    num_episodes = int(data["num_episodes"])

    episodes = []
    for i in range(num_episodes):
        ep = {}
        for key in ["real_frames", "imagined_frames", "context_recon", "context_real",
                     "actions", "rewards", "is_terminal", "episode_length", "context_length"]:
            full_key = f"ep{i}_{key}"
            if full_key in data:
                ep[key] = data[full_key]
        episodes.append(ep)

    return episodes


def compute_frame_metrics(real_frame, imagined_frame):
    """Compute MSE and SSIM between a pair of uint8 RGB frames."""
    real_f = real_frame.astype(np.float32) / 255.0
    imag_f = imagined_frame.astype(np.float32) / 255.0

    mse = float(np.mean((real_f - imag_f) ** 2))
    ssim_val = float(ssim(real_frame, imagined_frame, channel_axis=2, data_range=255))

    return {"mse": mse, "ssim": ssim_val}


def compute_semantic_metrics(real_frame, imagined_frame, extractor):
    """Extract game state from both frames and compare semantically."""
    real_state = extractor.extract(real_frame)
    imag_state = extractor.extract(imagined_frame)

    # If real frame has a head (it should), compare
    if real_state.head is not None:
        comparison = extractor.compare(
            real_state.head, real_state.body, real_state.food, imag_state
        )
    else:
        # Degenerate case — real frame doesn't have a head (shouldn't happen)
        comparison = {
            "head_error": 10.0,
            "body_accuracy": 0.0,
            "food_correct": False,
            "head_found": False,
            "food_found": False,
        }

    return comparison


def compute_episode_metrics(episode, extractor):
    """Compute all metrics for a single episode's imagination data."""
    real_frames = episode["real_frames"]
    imagined_frames = episode["imagined_frames"]
    n_steps = len(real_frames)

    per_step = []
    for t in range(n_steps):
        frame_met = compute_frame_metrics(real_frames[t], imagined_frames[t])
        sem_met = compute_semantic_metrics(real_frames[t], imagined_frames[t], extractor)

        step_metrics = {
            "step": t,
            "mse": frame_met["mse"],
            "ssim": frame_met["ssim"],
            "head_error": sem_met["head_error"],
            "body_accuracy": sem_met["body_accuracy"],
            "food_correct": bool(sem_met["food_correct"]),
            "head_found": bool(sem_met["head_found"]),
            "food_found": bool(sem_met["food_found"]),
        }
        per_step.append(step_metrics)

    return per_step


def aggregate_metrics(all_episode_metrics):
    """Aggregate per-step metrics across episodes into summary statistics."""
    metric_names = ["mse", "ssim", "head_error", "body_accuracy", "food_correct",
                    "head_found", "food_found"]

    # Overall aggregation
    all_values = {name: [] for name in metric_names}
    for ep_metrics in all_episode_metrics:
        for step in ep_metrics:
            for name in metric_names:
                all_values[name].append(float(step[name]))

    overall = {}
    for name in metric_names:
        vals = np.array(all_values[name])
        overall[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }

    # Per-step aggregation (average at each imagination step across episodes)
    max_steps = max(len(ep) for ep in all_episode_metrics)
    per_step_agg = []
    for t in range(max_steps):
        step_vals = {name: [] for name in metric_names}
        for ep_metrics in all_episode_metrics:
            if t < len(ep_metrics):
                for name in metric_names:
                    step_vals[name].append(float(ep_metrics[t][name]))

        step_agg = {"step": t, "n_episodes": len(step_vals["mse"])}
        for name in metric_names:
            vals = np.array(step_vals[name])
            step_agg[f"{name}_mean"] = float(np.mean(vals))
            step_agg[f"{name}_std"] = float(np.std(vals))
        per_step_agg.append(step_agg)

    # Early vs late horizon split (first 10 steps vs rest)
    early_cutoff = 10
    early_vals = {name: [] for name in metric_names}
    late_vals = {name: [] for name in metric_names}
    for ep_metrics in all_episode_metrics:
        for step in ep_metrics:
            bucket = early_vals if step["step"] < early_cutoff else late_vals
            for name in metric_names:
                bucket[name].append(float(step[name]))

    horizon = {}
    for label, vals in [("early", early_vals), ("late", late_vals)]:
        horizon[label] = {}
        for name in metric_names:
            if vals[name]:
                arr = np.array(vals[name])
                horizon[label][name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            else:
                horizon[label][name] = {"mean": None, "std": None}

    # Context reconstruction quality (how well does encoder→posterior→decode work)
    return {
        "overall": overall,
        "per_step": per_step_agg,
        "horizon": horizon,
        "num_episodes": len(all_episode_metrics),
        "total_frames": sum(len(ep) for ep in all_episode_metrics),
    }


def compute_checkpoint_metrics(imagination_npz_path, output_path=None):
    """Compute all metrics for a single checkpoint's imagination data.

    Args:
        imagination_npz_path: Path to imagination_data.npz
        output_path: Path to save metrics.json (default: same dir)

    Returns:
        Metrics dict
    """
    npz_path = pathlib.Path(imagination_npz_path)
    if output_path is None:
        output_path = npz_path.parent / "metrics.json"
    else:
        output_path = pathlib.Path(output_path)

    if output_path.exists():
        print(f"  Metrics already exist: {output_path}")
        with open(output_path) as f:
            return json.load(f)

    print(f"  Computing metrics for {npz_path.parent.name}...")
    episodes = load_imagination_data(npz_path)
    extractor = StateExtractor(grid_size=10, max_color_distance=100.0)

    all_episode_metrics = []
    for i, ep in enumerate(episodes):
        ep_metrics = compute_episode_metrics(ep, extractor)
        all_episode_metrics.append(ep_metrics)

    metrics = aggregate_metrics(all_episode_metrics)

    # Also store per-episode raw metrics for later analysis
    metrics["per_episode"] = []
    for i, ep_metrics in enumerate(all_episode_metrics):
        ep_summary = {
            "episode_idx": i,
            "num_steps": len(ep_metrics),
        }
        for name in ["mse", "ssim", "head_error", "body_accuracy", "food_correct"]:
            vals = [s[name] for s in ep_metrics]
            ep_summary[f"{name}_mean"] = float(np.mean(vals))
        metrics["per_episode"].append(ep_summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved metrics to {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute imagination metrics")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to imagination_data.npz")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for metrics.json")
    args = parser.parse_args()

    compute_checkpoint_metrics(args.input, args.output)


if __name__ == "__main__":
    main()
