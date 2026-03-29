"""Pure Snake game logic — no rendering, no external dependencies beyond numpy.

10x10 grid, 4 discrete actions, deterministic dynamics except food placement.
"""

from typing import NamedTuple, Optional
import numpy as np


class GameState(NamedTuple):
    """Complete snapshot of the game at one time step."""
    head: tuple[int, int]          # (row, col), 0-indexed, top-left is (0,0)
    body: list[tuple[int, int]]    # ordered from head to tail
    food: tuple[int, int]          # (row, col)
    direction: int                 # 0=up, 1=right, 2=down, 3=left
    score: int                     # food collected
    steps: int                     # steps taken
    done: bool                     # game over


# Direction vectors: (row_delta, col_delta)
DIRECTION_DELTAS = {
    0: (-1, 0),   # up
    1: (0, 1),    # right
    2: (1, 0),    # down
    3: (0, -1),   # left
}

# Opposite direction lookup (for preventing 180-degree turns)
OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}


class SnakeGame:
    """Snake game engine on an NxN grid.

    Actions: 0=up, 1=right, 2=down, 3=left.
    Reverse direction is ignored (no 180-degree turns).
    """

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self._rng: Optional[np.random.Generator] = None
        self.head = (0, 0)
        self.body: list[tuple[int, int]] = []
        self.food = (0, 0)
        self.direction = 1
        self.score = 0
        self.steps = 0
        self.done = False

    def reset(self, seed: Optional[int] = None) -> GameState:
        """Reset the game. Snake starts at center heading right, length 3."""
        self._rng = np.random.default_rng(seed)
        mid = self.grid_size // 2
        self.head = (mid, mid)
        # Body ordered head-first: head, then two trailing segments to the left
        self.body = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = 1  # heading right
        self.score = 0
        self.steps = 0
        self.done = False
        self.food = self._spawn_food()
        return self.get_state()

    def step(self, action: int) -> tuple[GameState, float, bool]:
        """Take one step. Returns (state, reward, done)."""
        if self.done:
            return self.get_state(), 0.0, True

        # Ignore reverse direction
        if action != OPPOSITE.get(self.direction, -1):
            self.direction = action

        # Compute new head position
        dr, dc = DIRECTION_DELTAS[self.direction]
        new_head = (self.head[0] + dr, self.head[1] + dc)

        # Check wall collision
        r, c = new_head
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            self.done = True
            self.steps += 1
            return self.get_state(), -1.0, True

        # Check self-collision (against current body minus tail, since tail
        # will move unless food is eaten — but we check before moving tail)
        # The tail will vacate its spot unless we eat food, so check against
        # body[:-1] (everything except the tail tip)
        if new_head in self.body[:-1]:
            self.done = True
            self.steps += 1
            return self.get_state(), -1.0, True

        # Move: insert new head
        self.head = new_head
        self.body.insert(0, new_head)

        # Check food
        reward = 0.0
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self.food = self._spawn_food()
            # Don't remove tail — snake grows
        else:
            self.body.pop()  # remove tail

        self.steps += 1
        return self.get_state(), reward, False

    def get_state(self) -> GameState:
        """Return a snapshot of the current game state."""
        return GameState(
            head=self.head,
            body=list(self.body),
            food=self.food,
            direction=self.direction,
            score=self.score,
            steps=self.steps,
            done=self.done,
        )

    def _spawn_food(self) -> tuple[int, int]:
        """Place food on a random empty cell."""
        occupied = set(self.body)
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in occupied
        ]
        if not empty:
            # Board is full — game effectively won
            return self.head  # placeholder
        idx = self._rng.integers(len(empty))
        return empty[idx]
