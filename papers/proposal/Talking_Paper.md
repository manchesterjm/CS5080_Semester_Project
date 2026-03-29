# Talking Paper — DreamerV3 Proposal

Print this. Glance at it, don't read it. The slides have the details — this is what you SAY.

---

## Slide 1: Title

"My project is about testing how accurate DreamerV3's imagination actually is, using Snake as the testbed."

---

## Slide 2: What is DreamerV3?

- Most RL agents learn by doing — trial and error. DreamerV3 flips that. It builds a mental model of the world first, then practices entirely inside its own head.
- Think of it like an athlete visualizing a play before running it. Except DreamerV3 *never* runs the play — it only visualizes.
- One algorithm, no tuning, works on everything from Atari to Minecraft. Published in Nature last year.

---

## Slide 3: Snake as MDP

- The agent sees raw pixels. A 64 by 64 image. It has no idea what a snake is — it has to figure that out from scratch.
- Movement is deterministic, only randomness is where food spawns. That's what makes it perfect for this — when the model gets something wrong, I know exactly what it got wrong and why.

---

## Slide 4: Baseline + The Problem

- Start with the obvious reward: plus one for food, minus one for death, zero otherwise. What does the agent do? It learns to survive. Not eat — survive.
- And honestly? That's the smart move. Why risk dying for one point when doing nothing is free? The expected value of seeking food can actually be *negative*.
- **Key line:** "The agent optimized exactly what we told it to — we just told it the wrong thing."
- Pause here. Let that land.

---

## Slide 5: Agent 2 — Step Penalty

- Fix is simple: charge rent. Minus 0.01 per step. Now sitting still bleeds points, so the agent *has* to go eat.
- This is the main agent for the imagination study. Forced food-seeking gives us richer trajectories to analyze.

---

## Slide 6: Agents 3–5

- Keep it quick. Same reward, different training configs. Short training, long training, different imagination horizons.
- The question across all five: does imagination accuracy actually track with performance?

---

## Slide 7: How DreamerV3 Reads the Game

- Pixels go into a CNN, get compressed into a latent space. A GRU tracks what's happening over time. A stochastic layer handles uncertainty.
- The actor and critic never see real pixels — they live entirely in latent imagination space.
- **Emphasize:** the model has to learn that a green blob IS the snake. Nobody told it that. That's why the semantic metrics matter.

---

## Slide 8: Measuring Imagination Accuracy

- Two levels. Pixel level — are the frames similar? Semantic level — does the model know where the snake actually is, where food is, when it's about to die?
- **The punchline:** a blurry dream that gets the positions right is fundamentally different from a sharp dream that puts the snake in the wrong place. We're measuring both.

---

## Slide 9: Timeline

- Walk through briefly. Baseline by end of Feb, primary agent early March, extraction pipeline mid-March, all agents done by end of March, midterm April 1.
- Everything runs on the 5070 Ti. Snake is small enough that compute isn't a concern.

---

## Slide 10: References

Skip — don't talk through this. It's there if someone asks.

---

## Slide 11: Questions

"Happy to take questions."

**If asked about the idle agent:** "Classic sparse reward problem. Well-known in RL. Step penalty is the standard fix."

**If asked about compute:** "Snake is tiny — 64 by 64, deterministic. Even 2 million steps is manageable on the 5070 Ti."

**If asked why five agents:** "Agent 1 showed the reward problem. Agent 2 fixes it. Agents 3 through 5 test whether training config affects imagination accuracy."

**If asked about generalization:** "This methodology works on any DreamerV3 environment, but Snake's simplicity makes it the ideal first test."
