# CS 5080 Semester Project: How Accurate Are Learned World Models?

An empirical analysis of DreamerV3's imagination fidelity on Snake.

**Author:** Josh Manchester
**Course:** CS 5080 Reinforcement Learning, Spring 2026, UCCS
**Instructor:** Dr. Jugal Kalita

## Research Question

Does DreamerV3 need accurate imagination to learn effective policies, or can it succeed even when its world model is wrong?

## Hypotheses

- **H1:** Imagination accuracy improves monotonically as training progresses
- **H2:** Better imagination correlates with better game performance
- **H3:** Semantic accuracy (game state) matters more than pixel-level accuracy

## Repository Structure

```
code/
  original/          # Pre-refactor source code (58 tests)
  refactored/        # Submission-quality code (121 tests, pylint 10/10, 88% coverage)
training_metrics/    # metrics.jsonl from each training run (lightweight)
analysis_output/     # Imagination data, computed metrics, hypothesis results per run
figures/             # Generated analysis figures (H1, H2, H3 plots)
papers/
  proposal/          # Proposal paper, presentation, talking paper
  midterm/           # Midterm paper, presentation, figures
video/
  trained_agent.mp4  # Demo of trained agent playing Snake
TRAINING_PLAN.md     # Complete model training history and hyperparameters
```

## Training Configurations

| Model | Config       | Steps | Key Change              |
|-------|-------------|-------|-------------------------|
| 2     | snake_1M    | 1.1M  | Baseline (64x64)        |
| 2     | snake_2M_long | 2M  | Extended baseline        |
| 2a    | snake_32x32 | 530K  | Resolution experiment    |
| 2b    | snake_16x16 | 514K  | Resolution experiment    |
| 4     | snake_high_lr | 2M  | 3x learning rate         |
| 5     | snake_low_lr | 2M   | 0.3x learning rate       |
| 6     | snake_high_tr | 2M  | 2x train ratio (in progress) |

## Environment

- **Game:** Snake on 10x10 grid, 64x64 RGB observations
- **Framework:** DreamerV3 (dreamerv3-torch)
- **GPU:** NVIDIA RTX 5070 Ti 16GB

## Note on Large Files

Model checkpoints (~210 MB each) and episode replay buffers (~154 GB total) are not included in this repository due to size constraints. They are backed up locally. The `training_metrics/` directory contains the `metrics.jsonl` logs which capture all training and evaluation statistics.
