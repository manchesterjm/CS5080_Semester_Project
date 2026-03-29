"""Gymnasium-compatible Snake environment with 64x64 RGB observations.

Color scheme (maximally distinct for state extraction):
    Background: (0, 0, 0)       black
    Snake head:  (0, 255, 0)    bright green
    Snake body:  (0, 128, 0)    dark green
    Food:        (255, 0, 0)    red

Grid mapping: 10x10 grid → 64x64 pixels.
Each cell = 6x6 pixels, with 2 pixels padding on each side.
Cell (r, c) occupies pixels [2+r*6 : 2+r*6+6, 2+c*6 : 2+c*6+6].
"""

from typing import Optional
import numpy as np
import gymnasium
from gymnasium import spaces

from snake_env.snake_game import SnakeGame
from constants import IMAGE_SIZE, CELL_SIZE, PADDING, STEP_PENALTY

# Colors (RGB)
COLOR_BG = (0, 0, 0)
COLOR_HEAD = (0, 255, 0)
COLOR_BODY = (0, 128, 0)
COLOR_FOOD = (255, 0, 0)


class SnakeEnv(gymnasium.Env):
    """Snake environment producing 64x64 RGB observations.

    Observation: (64, 64, 3) uint8 RGB image
    Action: Discrete(4) — 0=up, 1=right, 2=down, 3=left
    Reward: +1 food, -1 collision, -0.002 per step otherwise
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: int = 10,
        render_mode: str = "rgb_array",
        max_steps: int = 500,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.game = SnakeGame(grid_size)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

    def reset(  # pylint: disable=arguments-differ
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return initial observation and info."""
        super().reset(seed=seed)
        state = self.game.reset(seed=seed)
        obs = self._render_frame()
        info = {"game_state": state}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one step and return (obs, reward, terminated, truncated, info)."""
        state, reward, terminated = self.game.step(int(action))

        # Step penalty to discourage aimless circling
        if reward == 0:
            reward = STEP_PENALTY

        # Truncate if max steps reached
        truncated = (not terminated) and (self.game.steps >= self.max_steps)

        obs = self._render_frame()
        info = {"game_state": state}
        return obs, reward, terminated, truncated, info

    def _render_frame(self) -> np.ndarray:
        """Render current game state as a 64x64 RGB image."""
        frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        # Draw body segments (skip head, which is body[0])
        for r, c in self.game.body[1:]:
            self._fill_cell(frame, r, c, COLOR_BODY)

        # Draw head on top of body
        hr, hc = self.game.head
        self._fill_cell(frame, hr, hc, COLOR_HEAD)

        # Draw food
        fr, fc = self.game.food
        self._fill_cell(frame, fr, fc, COLOR_FOOD)

        return frame

    def _fill_cell(
        self, frame: np.ndarray, row: int, col: int, color: tuple[int, int, int]
    ) -> None:
        """Fill a grid cell's 6x6 pixel block with the given color."""
        y0 = PADDING + row * CELL_SIZE
        x0 = PADDING + col * CELL_SIZE
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = color

    def render(self) -> np.ndarray:
        """Return the current frame as an RGB array."""
        return self._render_frame()
