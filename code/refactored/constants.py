"""Shared constants for the DreamerV3 Snake project.

Centralizes magic numbers and repeated values to ensure consistency
across the game environment, analysis pipeline, and visualization scripts.
"""

# --- Game rewards and penalties ---
REWARD_FOOD = 1.0
REWARD_COLLISION = -1.0
STEP_PENALTY = -0.002
INITIAL_SNAKE_LENGTH = 3

# --- Grid and rendering ---
NUM_ACTIONS = 4
IMAGE_SIZE = 64
CELL_SIZE = 6
PADDING = 2

# --- Analysis ---
METRIC_NAMES = ("mse", "ssim", "head_error", "body_accuracy", "food_correct")
PIXEL_METRICS = ("mse", "ssim")
SEMANTIC_METRICS = ("head_error", "body_accuracy", "food_correct")
FIGURE_DPI = 150
P_VALUE_THRESHOLD = 0.05
CHECKPOINT_STEP_TOLERANCE = 20000

# --- Visualization / display ---
DISPLAY_SCALE = 8
DISPLAY_SIZE = IMAGE_SIZE * DISPLAY_SCALE
DEFAULT_FPS = 10
MAX_PLAY_STEPS = 100_000
