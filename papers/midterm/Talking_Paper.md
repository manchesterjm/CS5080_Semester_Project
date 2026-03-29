# Talking Paper — DreamerV3 Midterm Progress Report

Print this. Glance at it, don't read it. The slides have the details — this is what you SAY.

---

## Slide 1: Title

"This is a progress report on my DreamerV3 imagination fidelity project. Quick recap on the research question, then what I've found so far, and where things are headed."

---

## Slide 2: Research Question (Recap)

- "Quick recap from the proposal. DreamerV3 trains its policy entirely inside imagined trajectories — it never practices in the real environment."
- "The question is whether it needs those imaginations to be accurate, or whether it can succeed even with a wrong world model."
- "Three hypotheses: imagination improves over training, better imagination means better play, and game-state accuracy matters more than pixel accuracy."

---

## Slide 3: Why Snake (Recap)

- "Snake is the testbed because it's deterministic. When the model gets something wrong, I know exactly what went wrong."
- "Two levels of metrics: pixel level — do the images look similar? And semantic level — does the model actually know where the snake and food are?"
- "The key distinction: a blurry dream that gets the positions right is fundamentally different from a sharp dream that puts the snake in the wrong spot."

---

## Slide 4: What Has Been Completed

- "Five training configurations, all complete with full imagination analysis."
- "Each one changes one variable from the baseline — different learning rates, different observation resolutions, fine-tuned vs from-scratch."
- "Also trained a 3x learning rate model that never stabilized — the world model learned the game fine, but the policy kept overshooting. That's actually informative — a good world model doesn't guarantee a good policy."
- "Model 6 with doubled train ratio is currently running — about 40% done."

---

## Slide 5: Game Performance

- "Performance varies a lot across configs. Fine-tuned model with longer episodes peaks at 54. Lower resolutions top out around 9-13."
- "Key takeaway: resolution dramatically affects gameplay even though the underlying game is identical. The agent needs spatial detail to plan."

---

## Slide 6: H1 — Imagination Improves

- "H1 is well-supported. Models that are still actively learning show 4 or 5 out of 5 metrics significantly improving."
- "The Extended model — which inherited a good world model from fine-tuning — shows no significant improvement. The world model was already good."
- "This tells us something interesting: the world model learns fast, then plateaus. Policy learning is the slow part."
- "Head error in the baseline drops from 0.268 to 0.023 grid cells. That's almost perfect — it knows exactly where the snake head is."

---

## Slide 7: H2 — Imagination Correlates with Performance

- "H2 is also well-supported, but here's where it gets interesting."
- "Look at the Slow LR column. The pixel metrics — MSE and SSIM — are essentially at zero. They have no relationship with performance."
- "But the semantic metrics — body accuracy and head error — are strongly correlated."
- "Under the slow learning rate, image quality is completely decoupled from how well the agent plays."

---

## Slide 8: H3 — The Key Finding

- "This is the most exciting result so far. Under the slow learning rate, semantic metrics outpredict pixel metrics by more than 10x. Fisher's z-test is significant at p = 0.008."
- **Pause here.** "No other configuration reaches significance. The baseline is roughly tied. The resolution experiments are mixed."
- "Why does this only show up in the slow LR model? My interpretation: the reduced learning rate decouples visual reconstruction from semantic learning. Under normal rates, both improve together, so you can't tell which one the policy actually uses. The slow rate separates them."
- "The big question for the final paper is whether this replicates in other configurations. If it does, it's a general principle about model-based RL. If not, it might be specific to this training regime."

---

## Slide 9: Resolution Experiment

- "The resolution experiment is a useful control. Lower resolution means worse gameplay but cleaner statistical signals."
- "At 16x16, each grid cell is about 1.5 pixels — the world model can predict that easily, but the agent can't plan effectively."
- "This confirms the analysis pipeline works and the hypotheses behave as expected."

---

## Slide 10: Remaining Work

- "Model 6 should finish in about four days. Model 7 — larger board — is planned for the first two weeks of April."
- "I also want to add collision prediction accuracy as a metric, and test longer imagination horizons."
- "The key question: does the H3 result replicate?"

---

## Slide 11: Summary

- "H1 and H2 look solid. H3 has one significant result that needs replication."
- "The emerging picture: DreamerV3 doesn't need visually accurate dreams. It needs dreams that capture the right structure."
- "On track for the final paper."

---

## Slide 12: Questions

**If asked about the unstable 3x LR model:** "The world model learned the game — image loss was near zero. But the actor and critic at 3x learning rate kept overshooting gradient updates. Returns oscillated between -4 and +19 for the entire 2 million steps. A good world model isn't sufficient — you also need stable policy optimization."

**If asked why H3 is only significant in one model:** "Under normal learning rates, visual and semantic capabilities improve together. The slow rate decouples them, revealing that the policy relies on semantics. Whether that decoupling generalizes is the main question for the final paper."

**If asked about compute:** "Everything runs on an RTX 5070 Ti. Snake is small enough that compute isn't a concern — each 2M-step run takes about 3-4 days."

**If asked about generalization beyond Snake:** "The methodology works on any DreamerV3 environment, but Snake's simplicity makes it the ideal first test. The final paper could discuss how to extend the semantic extraction approach to more complex games."

**If asked about the step penalty:** "Without it, the agent learns to survive by doing nothing. The step penalty forces food-seeking, which gives richer trajectories for the imagination analysis."
