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
    """Get paths for a specific training run."""
    return {
        "checkpoint_dir": PROJECT_DIR / "training_output" / run_name / "checkpoints",
        "eval_dir": PROJECT_DIR / "training_output" / run_name / "eval_eps",
        "metrics_jsonl": PROJECT_DIR / "training_output" / run_name / "metrics.jsonl",
        "analysis_dir": PROJECT_DIR / f"analysis_output_{run_name}",
    }


def discover_checkpoints(checkpoint_arg, checkpoint_dir):
    """Parse checkpoint argument into list of checkpoint paths."""
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


def main():
    parser = argparse.ArgumentParser(description="Run full imagination analysis pipeline")
    parser.add_argument("--run", type=str, default="snake_1M",
                        help="Training run name (e.g. snake_1M, snake_2M_long)")
    parser.add_argument("--checkpoints", type=str, default="all",
                        help="'all' or comma-separated step numbers (e.g. 55000,105000)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes per checkpoint")
    parser.add_argument("--context", type=int, default=5,
                        help="Context length (observed frames before imagination)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Max imagination steps per episode")
    parser.add_argument("--skip-imagine", action="store_true",
                        help="Skip imagination step (reuse existing data)")
    parser.add_argument("--skip-metrics", action="store_true",
                        help="Skip metrics step (reuse existing metrics)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip final analysis step")
    parser.add_argument("--config", type=str, default=None,
                        help="Config section name from configs.yaml (default: same as --run)")
    args = parser.parse_args()
    if args.config is None:
        args.config = args.run

    paths = get_paths(args.run)
    CHECKPOINT_DIR = paths["checkpoint_dir"]
    EVAL_DIR = paths["eval_dir"]
    METRICS_JSONL = paths["metrics_jsonl"]
    ANALYSIS_DIR = paths["analysis_dir"]

    checkpoint_paths = discover_checkpoints(args.checkpoints, CHECKPOINT_DIR)
    print(f"Run: {args.run}")
    print(f"Found {len(checkpoint_paths)} checkpoints to process")
    print(f"Output: {ANALYSIS_DIR}")
    print(f"Settings: episodes={args.episodes}, context={args.context}, "
          f"max_steps={args.max_steps}, device={args.device}")
    print()

    total_start = time.time()

    # Stage 1: Imagination extraction
    if not args.skip_imagine:
        print("=" * 60)
        print("STAGE 1: Imagination Extraction")
        print("=" * 60)
        for i, ckpt_path in enumerate(checkpoint_paths):
            ckpt_name = ckpt_path.stem
            output_dir = ANALYSIS_DIR / ckpt_name
            print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_name}")

            start = time.time()
            run_imagination(
                checkpoint_path=str(ckpt_path),
                eval_dir=str(EVAL_DIR),
                output_dir=str(output_dir),
                num_episodes=args.episodes,
                context_length=args.context,
                device=args.device,
                max_imagine_steps=args.max_steps,
                config_name=args.config,
            )
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.1f}s")
    else:
        print("Skipping imagination extraction (--skip-imagine)")

    # Stage 2: Metric computation
    if not args.skip_metrics:
        print("\n" + "=" * 60)
        print("STAGE 2: Metric Computation")
        print("=" * 60)
        for i, ckpt_path in enumerate(checkpoint_paths):
            ckpt_name = ckpt_path.stem
            npz_path = ANALYSIS_DIR / ckpt_name / "imagination_data.npz"
            if not npz_path.exists():
                print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_name} — SKIPPED (no imagination data)")
                continue

            print(f"\n[{i + 1}/{len(checkpoint_paths)}] {ckpt_name}")
            start = time.time()
            compute_checkpoint_metrics(str(npz_path))
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.1f}s")
    else:
        print("Skipping metric computation (--skip-metrics)")

    # Stage 3: Cross-checkpoint analysis
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("STAGE 3: Cross-Checkpoint Analysis")
        print("=" * 60)
        run_analysis(str(ANALYSIS_DIR), str(METRICS_JSONL))
    else:
        print("Skipping analysis (--skip-analysis)")

    total_elapsed = time.time() - total_start
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
