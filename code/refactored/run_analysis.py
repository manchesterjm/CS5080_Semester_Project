"""Orchestration script for the full imagination analysis pipeline.

Runs all three stages across all checkpoints:
1. imagine.py  — extract imagined trajectories
2. compute_metrics.py — compute per-checkpoint metrics
3. analyze.py — cross-checkpoint hypothesis testing

Usage:
    python run_analysis.py [--checkpoints all|55000,105000,...] [--episodes 20] [--context 5]
"""

import argparse
import pathlib
import time

from imagine import run_imagination
from compute_metrics import compute_checkpoint_metrics
from analyze import run_analysis


PROJECT_DIR = pathlib.Path(__file__).parent


def get_paths(run_name):
    """Return a dict of checkpoint, eval, metrics, and output paths for a run."""
    return {
        "checkpoint_dir": PROJECT_DIR / "training_output" / run_name / "checkpoints",
        "eval_dir": PROJECT_DIR / "training_output" / run_name / "eval_eps",
        "metrics_jsonl": PROJECT_DIR / "training_output" / run_name / "metrics.jsonl",
        "analysis_dir": PROJECT_DIR / f"analysis_output_{run_name}",
    }


def discover_checkpoints(checkpoint_arg, checkpoint_dir):
    """Parse 'all' or comma-separated steps into sorted checkpoint paths."""
    if checkpoint_arg == "all":
        paths = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    else:
        steps = [int(s.strip()) for s in checkpoint_arg.split(",")]
        paths = []
        for step in steps:
            path = checkpoint_dir / f"checkpoint_{step:07d}.pt"
            if path.exists():
                paths.append(path)
            else:
                print(f"WARNING: Checkpoint not found: {path}")
    return paths


def _parse_args():
    """Parse command-line arguments for the analysis pipeline."""
    parser = argparse.ArgumentParser(description="Run full imagination analysis pipeline")
    parser.add_argument("--run", type=str, default="snake_1M",
                        help="Training run name (e.g. snake_1M, snake_2M_long)")
    parser.add_argument("--checkpoints", type=str, default="all",
                        help="'all' or comma-separated step numbers")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--skip-imagine", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    if args.config is None:
        args.config = args.run
    return args


def _run_imagination_stage(checkpoint_paths, args, paths):
    """Stage 1: Extract imagined trajectories from each checkpoint."""
    print("=" * 60)
    print("STAGE 1: Imagination Extraction")
    print("=" * 60)
    for i, ckpt_path in enumerate(checkpoint_paths):
        ckpt_name = ckpt_path.stem
        output_dir = paths["analysis_dir"] / ckpt_name
        print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_name}")
        start = time.time()
        run_imagination(
            checkpoint_path=str(ckpt_path),
            eval_dir=str(paths["eval_dir"]),
            output_dir=str(output_dir),
            num_episodes=args.episodes,
            context_length=args.context,
            device=args.device,
            max_imagine_steps=args.max_steps,
            config_name=args.config,
        )
        print(f"  Time: {time.time() - start:.1f}s")


def _run_metrics_stage(checkpoint_paths, paths):
    """Stage 2: Compute per-checkpoint metrics from imagination data."""
    print("\n" + "=" * 60)
    print("STAGE 2: Metric Computation")
    print("=" * 60)
    for i, ckpt_path in enumerate(checkpoint_paths):
        ckpt_name = ckpt_path.stem
        npz_path = paths["analysis_dir"] / ckpt_name / "imagination_data.npz"
        if not npz_path.exists():
            print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_name} — SKIPPED")
            continue
        print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_name}")
        start = time.time()
        compute_checkpoint_metrics(str(npz_path))
        print(f"  Time: {time.time() - start:.1f}s")


def main():
    """Parse arguments and run the three-stage analysis pipeline."""
    args = _parse_args()
    paths = get_paths(args.run)
    checkpoint_paths = discover_checkpoints(args.checkpoints, paths["checkpoint_dir"])
    print(f"Run: {args.run}")
    print(f"Found {len(checkpoint_paths)} checkpoints to process")
    print(f"Output: {paths['analysis_dir']}")
    print(f"Settings: episodes={args.episodes}, context={args.context}, "
          f"max_steps={args.max_steps}, device={args.device}\n")

    total_start = time.time()
    if not args.skip_imagine:
        _run_imagination_stage(checkpoint_paths, args, paths)
    else:
        print("Skipping imagination extraction (--skip-imagine)")
    if not args.skip_metrics:
        _run_metrics_stage(checkpoint_paths, paths)
    else:
        print("Skipping metric computation (--skip-metrics)")
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("STAGE 3: Cross-Checkpoint Analysis")
        print("=" * 60)
        run_analysis(str(paths["analysis_dir"]), str(paths["metrics_jsonl"]))
    else:
        print("Skipping analysis (--skip-analysis)")
    elapsed = time.time() - total_start
    print(f"\nTotal pipeline time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
