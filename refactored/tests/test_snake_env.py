"""Tests for SnakeEnv Gymnasium wrapper."""

import numpy as np
import pytest
from snake_env.snake_env import SnakeEnv, IMAGE_SIZE, CELL_SIZE, PADDING


class TestSnakeEnvAPI:
    """Test Gymnasium API compliance."""

    def test_observation_space_shape(self):
        env = SnakeEnv()
        assert env.observation_space.shape == (64, 64, 3)
        assert env.observation_space.dtype == np.uint8

    def test_action_space_size(self):
        env = SnakeEnv()
        assert env.action_space.n == 4

    def test_reset_returns_obs_and_info(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        assert "game_state" in info

    def test_step_returns_five_tuple(self):
        env = SnakeEnv()
        env.reset(seed=42)
        result = env.step(1)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "game_state" in info

    def test_obs_within_bounds(self):
        env = SnakeEnv()
        obs, _ = env.reset(seed=42)
        assert obs.min() >= 0
        assert obs.max() <= 255


class TestSnakeEnvRendering:
    """Test that rendering produces correct pixel output."""

    def test_frame_is_mostly_black(self):
        env = SnakeEnv()
        obs, _ = env.reset(seed=42)
        # Most pixels should be background (black)
        black_pixels = np.all(obs == 0, axis=2)
        # At least 80% should be black (snake=3 cells, food=1 cell, out of 100)
        assert black_pixels.sum() > 0.8 * 64 * 64

    def test_head_is_bright_green(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)
        head_r, head_c = info["game_state"].head
        # Check center pixel of head cell
        cy = PADDING + head_r * CELL_SIZE + CELL_SIZE // 2
        cx = PADDING + head_c * CELL_SIZE + CELL_SIZE // 2
        pixel = obs[cy, cx]
        assert tuple(pixel) == (0, 255, 0)

    def test_body_is_dark_green(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)
        # Second body segment (not head)
        body_r, body_c = info["game_state"].body[1]
        cy = PADDING + body_r * CELL_SIZE + CELL_SIZE // 2
        cx = PADDING + body_c * CELL_SIZE + CELL_SIZE // 2
        pixel = obs[cy, cx]
        assert tuple(pixel) == (0, 128, 0)

    def test_food_is_red(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)
        food_r, food_c = info["game_state"].food
        cy = PADDING + food_r * CELL_SIZE + CELL_SIZE // 2
        cx = PADDING + food_c * CELL_SIZE + CELL_SIZE // 2
        pixel = obs[cy, cx]
        assert tuple(pixel) == (255, 0, 0)

    def test_cell_is_uniform_color(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)
        head_r, head_c = info["game_state"].head
        y0 = PADDING + head_r * CELL_SIZE
        x0 = PADDING + head_c * CELL_SIZE
        cell = obs[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]
        # All pixels in the cell should be the same color
        assert np.all(cell == cell[0, 0])


class TestSnakeEnvTruncation:
    """Test max step truncation."""

    def test_truncation_at_max_steps(self):
        env = SnakeEnv(max_steps=10)
        env.reset(seed=42)
        for i in range(20):
            obs, reward, terminated, truncated, info = env.step(0)  # up
            if terminated or truncated:
                break
        assert terminated or truncated
        # If it hit a wall first, that's terminated; if it survived 10 steps, truncated


class TestSnakeEnvGameState:
    """Test that game state is properly exposed in info dict."""

    def test_info_has_game_state(self):
        env = SnakeEnv()
        obs, info = env.reset(seed=42)
        gs = info["game_state"]
        assert hasattr(gs, "head")
        assert hasattr(gs, "body")
        assert hasattr(gs, "food")
        assert hasattr(gs, "score")
        assert hasattr(gs, "done")

    def test_game_state_updates_on_step(self):
        env = SnakeEnv()
        obs, info1 = env.reset(seed=42)
        obs, _, _, _, info2 = env.step(1)
        assert info2["game_state"].steps == 1
        assert info2["game_state"].head != info1["game_state"].head


class TestSnakeEnvMultipleEpisodes:
    """Test running multiple episodes."""

    def test_ten_episodes(self):
        env = SnakeEnv(max_steps=100)
        for ep in range(10):
            obs, info = env.reset(seed=ep)
            assert obs.shape == (64, 64, 3)
            done = False
            steps = 0
            while not done and steps < 150:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                assert obs.shape == (64, 64, 3)
