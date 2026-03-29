# DreamerV3 Snake — Training Plan

## Model History

### Model 1: Baseline (No Step Penalty)
- **Result:** Agent learned to circle endlessly for 500 steps, exploiting the lack of penalty for inaction
- **Lesson:** Without a step penalty, the agent has no incentive to seek food — surviving is "free"

### Model 2: Step Penalty Added
- **Config:** 10x10 grid, 500 max steps, 1M training steps, 4 envs, train ratio 256
- **Step penalty:** -0.002 per step
- **Result:** Agent learned to play snake. ~1.1M steps completed, avg return 19.65, peak ~29.1.
- **Eval (no step limit):** Scores 21-36, avg ~31. All deaths from self-collision — agent navigates well with short body but struggles once body length exceeds ~30 on the 10x10 grid.
- **Lesson:** 500-step cap prevents agent from ever training on long-body navigation. It never sees states where >1/3 of the board is occupied.
- **Output:** `training_output/snake_1M/`
- **Key hyperparameters:**

| Parameter    | Value  |
| ------------ | ------ |
| Grid         | 10x10  |
| Observation  | 64x64  |
| Max steps/ep | 500    |
| Training steps | 1M   |
| Parallel envs | 4     |
| Train ratio  | 256    |
| Model LR     | 1e-4   |
| Actor LR     | 3e-5   |
| Critic LR    | 3e-5   |
| Discount     | 0.999  |
| Step penalty | -0.002 |
| Batch size   | 16     |
| Batch length | 64     |
| Imag horizon | 15     |
| Prefill      | 5000   |

### Model 3: Longer Episodes — Fine-tuned from Model 2 (Current Run)
- **Approach:** Fine-tune from Model 2 checkpoint (not trained from scratch)
- **Changes from Model 2:** max steps 500→2000, training steps 1M→2M, envs 4→8
- **Rationale:** With 2000 steps, the agent will regularly encounter long-body states (50+ cells) and must learn to navigate tight spaces. Fine-tuning preserves Model 2's existing snake skills while extending to the harder long-body task.
- **Output:** `training_output/snake_2M_long/`
- **Key hyperparameters:**

| Parameter      | Value  |
| -------------- | ------ |
| Grid           | 10x10  |
| Observation    | 64x64  |
| Max steps/ep   | 2000   |
| Training steps | 2M     |
| Parallel envs  | 8      |
| Train ratio    | 256    |
| Model LR       | 1e-4   |
| Actor LR       | 3e-5   |
| Critic LR      | 3e-5   |
| Discount       | 0.999  |
| Step penalty   | -0.002 |
| Batch size     | 16     |
| Batch length   | 64     |
| Imag horizon   | 15     |
| Prefill        | 5000   |

## Resolution Experiment (Completed 2026-03-04)

Tested whether observation resolution affects world model quality and gameplay.
All resolution variants use same 10x10 grid, 2000 max steps, 8 envs, train ratio 512.

**NOTE (2026-03-29):** The `snake_32x32` and `snake_16x16` configs were mistakenly set to `steps: 5e5` in `configs.yaml` instead of `2e6`. Training completed at ~557K and ~514K steps respectively — these are NOT paused runs, they hit their (incorrect) step target and stopped. The standard target for all models is 2M. This limits comparability with the 2M runs. Too late to re-run for the midterm; results are presented as-is with this caveat noted.

### Configurations

| Model     | Obs Res | Steps   | Envs | Train Ratio | Batch Size | Notes                        |
| --------- | ------- | ------- | ---- | ----------- | ---------- | ---------------------------- |
| snake_1M  | 64x64   | 1.1M    | 4    | 256         | 16         | Original baseline            |
| 2M_long   | 64x64   | 2M      | 8    | 256         | 16         | Fine-tuned from snake_1M     |
| snake_32  | 32x32   | ~557K   | 10   | 512         | 32         | Config error: target was 500K |
| snake_16  | 16x16   | ~514K   | 10   | 512         | 32         | Config error: target was 500K |

### Game Performance

| Model     | Mean Return | Peak Return | Final 5 Avg |
| --------- | ----------- | ----------- | ----------- |
| 2M_long   | 31.17       | 54.26       | 36.50       |
| snake_1M  | 9.82        | 25.71       | 22.28       |
| snake_32  | 1.06        | 13.60       | 9.83        |
| snake_16  | 0.78        | 9.13        | 7.44        |

### Hypothesis Results Summary

**H1 — Imagination improves over training:**

| Model    | Improving | Significant |
| -------- | --------- | ----------- |
| snake_16 | 5/5       | 5/5         |
| snake_32 | 5/5       | 5/5         |
| snake_1M | 5/5       | 4/5         |
| 2M_long  | 3/5       | 0/5         |

**H2 — Imagination correlates with performance (Spearman |rho|):**

| Metric        | 16x16 | 32x32 | 1M (64x64) | 2M_long |
| ------------- | ----- | ----- | ---------- | ------- |
| MSE           | 0.564 | 0.603 | 0.570      | 0.099   |
| SSIM          | 0.641 | 0.660 | 0.630      | 0.120   |
| Head Error    | 0.678 | 0.513 | 0.843      | 0.163   |
| Body Accuracy | 0.532 | 0.292 | 0.352      | 0.511   |
| Food Correct  | 0.636 | 0.654 | 0.597      | 0.090   |

**H3 — Semantic vs pixel metrics:**

| Model    | Pixel Avg |rho| | Semantic Avg |rho| | Advantage       |
| -------- | --------------- | ------------------- | --------------- |
| snake_16 | 0.603           | 0.615               | Semantic +0.013 |
| snake_32 | 0.631           | 0.486               | Pixel +0.145    |
| snake_1M | 0.600           | 0.597               | ~Tied           |
| 2M_long  | 0.110           | 0.255               | Semantic +0.145 |

Fisher's z-test not significant for any model (p > 0.15 for all).

### Conclusions

1. **Resolution strongly impacts gameplay.** 64x64 dramatically outperforms lower resolutions despite same grid size. The agent needs spatial detail to plan effectively.
2. **Lower resolutions produce cleaner learning signals.** 16x16 and 32x32 show 5/5 significant H1 trends — the world model steadily improves. The 2M_long model shows flat imagination (already had a good world model from fine-tuning).
3. **H3 is inconclusive.** No consistent winner between pixel and semantic metrics across resolutions. Direction flips between models.
4. **16x16 vs 32x32:** Very similar patterns. 16x16 has slightly stronger H2 correlations (simpler visual task makes world modeling easier) but worse gameplay (less spatial info for planning).

### Analysis Outputs

| Model    | Directory                        |
| -------- | -------------------------------- |
| snake_1M | `analysis_output/`               |
| 2M_long  | `analysis_output_snake_2M_long/` |
| snake_32 | `analysis_output_snake_32x32/`   |
| snake_16 | `analysis_output_snake_16x16/`   |

---

## Way Ahead

All future models use **8 parallel environments** and **2000 max steps** as the new baseline.

### Model 4: Higher Learning Rate (Complete)
- **Change:** model_lr 3e-4, actor/critic lr 1e-4 (3x faster across the board)
- **Tests:** Does faster learning speed convergence, or cause instability?
- **Output:** `training_output/snake_high_lr/`
- **Final checkpoint:** `checkpoint_2011006.pt` (2M steps, completed 2026-03-19 overnight)
- **Results: Consistently poor with high variance.** Eval returns oscillate wildly between -4.0 and +4.4 with no stable upward trend. Best eval: 4.4 at 1,055K. Most evals hover around -1 to -2. Compare to baseline which was reliably scoring 20+ by this point.
- **Eval return trajectory (sampled):**
  - 0-75K: -0.8 to -4.0 (random)
  - 75K-420K: mostly -2 to -4 (stuck)
  - 420K-640K: -3.7 to -0.4 (slight improvement)
  - 640K-960K: oscillating -3.6 to +2.7 (high variance)
  - 960K-1,275K: oscillating -3.8 to +4.4 (still high variance, no convergence)
  - 1,275K-2M: continued oscillation (19.0, 18.8, -0.0, 5.9 in final logs) — no convergence
- **World model is fine:** image_loss ~0.0, model_loss ~1.0 — the world model learned the game
- **Policy is unstable:** Actor/critic at 3x LR overshoots gradient updates. Returns swing wildly per eval, never stabilizing. Baseline's actor/critic at 3e-5 produced smooth, consistent improvement.
- **Conclusion:** Higher LR destabilizes policy learning despite adequate world model quality. The agent occasionally stumbles into good behavior but can't consolidate it. 3x LR is too aggressive for this task.

### Model 5: Lower Learning Rate (Complete)
- **Change:** model_lr 3e-5, actor/critic lr 1e-5 (3x slower across the board)
- **Tests:** More stable learning? Or underfitting within 2M steps?
- **Output:** `training_output/snake_low_lr/`
- **Final checkpoint:** `checkpoint_2010324.pt` (2M steps, completed 2026-03-25 ~07:30 MDT)
- **Run history:** Trained to 195K in prior session, resumed 2026-03-19 ~22:37 MDT, paused 2026-03-20 ~06:57 MDT at 551K steps (~8h 17m runtime that session). Resumed again, reached ~1,006K/2M as of 2026-03-21. Windows Update auto-restart killed training at ~1,070K on 2026-03-22. Resumed 2026-03-22 ~22:15 MDT, paused 2026-03-23 ~04:00 MDT at ~1,325K. Resumed 2026-03-24 ~22:27 MDT, paused ~05:33 MDT at ~1,978K. Resumed ~05:45 MDT, completed ~07:30 MDT on 2026-03-25.
- **Training rate:** 12.5 FPS avg (11.72 steps/sec wall clock including eval pauses)
- **Results: Slow start but competitive by 2M.** Mean eval return 16.57 across 75 evals. Peak eval: 24.6. Last 5 avg: 19.16. Last 10 avg: 18.16.
- **Eval return trajectory (sampled):**
  - 200K-290K: oscillating -2.9 to +0.9 (first 12 avg: -1.35)
  - 300K-430K: mostly -2.0 to -3.6 (worse phase)
  - 450K-551K: slight improvement, last 12 avg: -0.86 (best: +0.1)
  - 551K-1,006K: breakthrough — returns reaching 12.3, 12.5 range
  - 1,070K-1,325K: 10-17 range, consolidating
  - 1,325K-2M: 13-24 range, final eval returns: 23.9, 17.6, 19.8, 16.1
- **Comparison to baseline:** Baseline (Model 3) mean eval 31.17, peak 54.26. Model 5 is noticeably weaker (mean 16.57, peak 24.6) — 3x slower LR couldn't converge to the same level within 2M steps, though it was still improving at end of training.
- **Conclusion:** 3x slower LR severely underfits early on (negative returns through 500K), then recovers to competitive but inferior performance. The agent learned to play but never matched baseline quality. Policy learning was too slow to fully exploit what the world model learned. Would likely improve with more training steps, but 2M wasn't enough.

#### Model 5: Analysis Results (2026-03-25)
- **21 checkpoints analyzed** (every ~100K from 55K to 2M)
- **H1 — Imagination improves over training:** 5/5 improving, 2/5 significant (body_accuracy p=0.0000, head_error p=0.003)
- **H2 — Imagination correlates with performance:**

| Metric        | Spearman |rho| | p-value  |
| ------------- | --------------- | -------- |
| MSE           | 0.023           | 0.920    |
| SSIM          | 0.066           | 0.776    |
| Head Error    | 0.514           | 0.017*   |
| Body Accuracy | 0.740           | 0.0001*  |
| Food Correct  | 0.310           | 0.171    |

- **H3 — Semantic vs pixel metrics:** Semantic avg |rho|=0.522 vs Pixel avg |rho|=0.045. **Fisher's z-test: p=0.008 (significant!)**
- **Notable:** First model to achieve a statistically significant H3 result. Semantic metrics (especially body accuracy) strongly outpredict pixel metrics for game performance. The slower learning rate produces a cleaner signal where world model quality directly drives agent behavior.

### Model 6: Higher Train Ratio (Config Ready)
- **Change:** train_ratio 512 (up from 256)
- **Tests:** More dream steps per real environment step — does heavier imagination help?
- **Config:** `snake_high_tr` in configs.yaml
- **Output:** `training_output/snake_high_tr/`

### Model 7: Large Board
- **Change:** 20x20 grid, 128x128 observations, 2-3M training steps
- **Tests:** Does the world model scale to a harder planning problem?
- **Purpose:** Visual demo — watch the agent play on a real-sized board

## Comparison Methodology

- Each model changes **one variable** from the current baseline (except Model 7 which is the scale test)
- Compare: peak return, average eval return, convergence speed, consistency (variance in returns)
- All runs logged to `training_output/` with separate directories per model
