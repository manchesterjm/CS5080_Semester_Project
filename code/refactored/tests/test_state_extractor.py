"""Tests for StateExtractor — parsing RGB frames back to game state."""

import numpy as np
import pytest
from snake_env.snake_env import SnakeEnv
from snake_env.state_extractor import StateExtractor


class TestStateExtractorRealFrames:
    """Test extraction from crisp, real-rendered frames (should be 100% accurate)."""

    def test_head_position_exact(self):
        env = SnakeEnv()
        extractor = StateExtractor()
        obs, info = env.reset(seed=42)
        extracted = extractor.extract(obs)
        assert extracted.head == info["game_state"].head

    def test_body_positions_exact(self):
        env = SnakeEnv()
        extractor = StateExtractor()
        obs, info = env.reset(seed=42)
        extracted = extractor.extract(obs)
        real_body_no_head = info["game_state"].body[1:]
        assert set(extracted.body) == set(real_body_no_head)

    def test_food_position_exact(self):
        env = SnakeEnv()
        extractor = StateExtractor()
        obs, info = env.reset(seed=42)
        extracted = extractor.extract(obs)
        assert extracted.food == info["game_state"].food

    def test_round_trip_after_multiple_steps(self):
        env = SnakeEnv()
        extractor = StateExtractor()
        env.reset(seed=42)
        actions = [1, 1, 0, 0, 3, 2, 2, 1]
        for action in actions:
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            extracted = extractor.extract(obs)
            gs = info["game_state"]
            assert extracted.head == gs.head, f"Head mismatch at step"
            assert extracted.food == gs.food, f"Food mismatch at step"
            real_body_no_head = gs.body[1:]
            assert set(extracted.body) == set(real_body_no_head)

    def test_compare_returns_perfect_scores(self):
        env = SnakeEnv()
        extractor = StateExtractor()
        obs, info = env.reset(seed=42)
        extracted = extractor.extract(obs)
        gs = info["game_state"]
        metrics = extractor.compare(
            gs.head, gs.body[1:], gs.food, extracted
        )
        assert metrics["head_error"] == 0.0
        assert metrics["body_accuracy"] == 1.0
        assert metrics["food_correct"] is True
        assert metrics["head_found"] is True
        assert metrics["food_found"] is True


class TestStateExtractorBlurryFrames:
    """Test extraction from simulated blurry/noisy frames."""

    def test_tolerates_slight_noise(self):
        env = SnakeEnv()
        extractor = StateExtractor(max_color_distance=50.0)
        obs, info = env.reset(seed=42)
        # Add small Gaussian noise
        noisy = obs.astype(np.float32) + np.random.RandomState(0).normal(0, 10, obs.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        extracted = extractor.extract(noisy)
        # Should still identify head and food correctly with small noise
        assert extracted.head == info["game_state"].head
        assert extracted.food == info["game_state"].food

    def test_label_grid_shape(self):
        env = SnakeEnv()
        extractor = StateExtractor()
        obs, _ = env.reset(seed=42)
        extracted = extractor.extract(obs)
        assert extracted.cell_labels.shape == (10, 10)


class TestStateExtractorMultipleSeeds:
    """Test extraction across different game configurations."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 123, 456, 789])
    def test_perfect_extraction_various_seeds(self, seed):
        env = SnakeEnv()
        extractor = StateExtractor()
        obs, info = env.reset(seed=seed)
        extracted = extractor.extract(obs)
        gs = info["game_state"]
        assert extracted.head == gs.head
        assert extracted.food == gs.food
        assert set(extracted.body) == set(gs.body[1:])
