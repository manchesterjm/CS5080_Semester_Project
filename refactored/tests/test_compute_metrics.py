"""Tests for compute_metrics module."""

import json
import tempfile
import pathlib

import numpy as np
import pytest

from compute_metrics import (
    compute_frame_metrics,
    compute_semantic_metrics,
    compute_episode_metrics,
    aggregate_metrics,
    compute_checkpoint_metrics,
    load_imagination_data,
    _aggregate_overall,
    _aggregate_per_step,
    _aggregate_horizon,
    EXTENDED_METRICS,
    EARLY_CUTOFF,
)
from snake_env.state_extractor import StateExtractor


class TestFrameMetrics:
    """Tests for compute_frame_metrics."""

    def test_identical_frames(self):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = compute_frame_metrics(frame, frame)
        assert result["mse"] == pytest.approx(0.0, abs=1e-6)
        assert result["ssim"] == pytest.approx(1.0, abs=1e-3)

    def test_different_frames(self):
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = compute_frame_metrics(black, white)
        assert result["mse"] == pytest.approx(1.0, abs=1e-6)
        assert result["ssim"] < 0.1

    def test_returns_float_values(self):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = compute_frame_metrics(frame, frame)
        assert isinstance(result["mse"], float)
        assert isinstance(result["ssim"], float)


class TestSemanticMetrics:
    """Tests for compute_semantic_metrics."""

    def test_identical_real_frame(self):
        """A real frame compared to itself should give perfect scores."""
        from snake_env.snake_env import SnakeEnv
        env = SnakeEnv()
        obs, _ = env.reset(seed=42)
        extractor = StateExtractor(grid_size=10, max_color_distance=100.0)
        result = compute_semantic_metrics(obs, obs, extractor)
        assert result["head_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["body_accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert result["food_correct"] is True
        assert result["head_found"] is True

    def test_completely_different_frame(self):
        """A black frame should fail to find game objects."""
        from snake_env.snake_env import SnakeEnv
        env = SnakeEnv()
        obs, _ = env.reset(seed=42)
        black = np.zeros_like(obs)
        extractor = StateExtractor(grid_size=10, max_color_distance=100.0)
        result = compute_semantic_metrics(obs, black, extractor)
        assert result["head_found"] is False
        assert result["food_found"] is False


class TestEpisodeMetrics:
    """Tests for compute_episode_metrics."""

    def test_returns_per_step_list(self):
        from snake_env.snake_env import SnakeEnv
        env = SnakeEnv()
        obs, _ = env.reset(seed=42)
        episode = {
            "real_frames": np.stack([obs, obs, obs]),
            "imagined_frames": np.stack([obs, obs, obs]),
        }
        extractor = StateExtractor(grid_size=10, max_color_distance=100.0)
        result = compute_episode_metrics(episode, extractor)
        assert len(result) == 3
        assert all("mse" in step for step in result)
        assert all("ssim" in step for step in result)
        assert all("step" in step for step in result)


class TestAggregation:
    """Tests for aggregation functions."""

    def _make_step(self, step_num, mse=0.1):
        return {
            "step": step_num, "mse": mse, "ssim": 0.9,
            "head_error": 0.5, "body_accuracy": 0.8, "food_correct": True,
            "head_found": True, "food_found": True,
        }

    def test_aggregate_overall_basic(self):
        episodes = [[self._make_step(0), self._make_step(1)]]
        result = _aggregate_overall(episodes)
        assert "mse" in result
        assert result["mse"]["mean"] == pytest.approx(0.1, abs=1e-6)

    def test_aggregate_per_step(self):
        episodes = [
            [self._make_step(0, mse=0.1), self._make_step(1, mse=0.2)],
            [self._make_step(0, mse=0.3), self._make_step(1, mse=0.4)],
        ]
        result = _aggregate_per_step(episodes)
        assert len(result) == 2
        assert result[0]["mse_mean"] == pytest.approx(0.2, abs=1e-6)
        assert result[1]["mse_mean"] == pytest.approx(0.3, abs=1e-6)

    def test_aggregate_horizon_early_late(self):
        early_steps = [self._make_step(i, mse=0.1) for i in range(EARLY_CUTOFF)]
        late_steps = [self._make_step(i + EARLY_CUTOFF, mse=0.5)
                      for i in range(5)]
        episodes = [early_steps + late_steps]
        result = _aggregate_horizon(episodes)
        assert result["early"]["mse"]["mean"] == pytest.approx(0.1, abs=1e-6)
        assert result["late"]["mse"]["mean"] == pytest.approx(0.5, abs=1e-6)

    def test_aggregate_metrics_full(self):
        episodes = [[self._make_step(0), self._make_step(1)]]
        result = aggregate_metrics(episodes)
        assert "overall" in result
        assert "per_step" in result
        assert "horizon" in result
        assert result["num_episodes"] == 1
        assert result["total_frames"] == 2


class TestLoadImaginationData:
    """Tests for load_imagination_data."""

    def test_load_round_trip(self, tmp_path):
        num_eps = 2
        save_dict = {"num_episodes": np.array(num_eps)}
        for i in range(num_eps):
            save_dict[f"ep{i}_real_frames"] = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            save_dict[f"ep{i}_imagined_frames"] = np.zeros((3, 64, 64, 3), dtype=np.uint8)
            save_dict[f"ep{i}_context_recon"] = np.zeros((2, 64, 64, 3), dtype=np.uint8)
            save_dict[f"ep{i}_context_real"] = np.zeros((2, 64, 64, 3), dtype=np.uint8)
            save_dict[f"ep{i}_actions"] = np.zeros((3, 4), dtype=np.float32)
            save_dict[f"ep{i}_rewards"] = np.zeros(3, dtype=np.float32)
            save_dict[f"ep{i}_is_terminal"] = np.zeros(3, dtype=bool)
            save_dict[f"ep{i}_episode_length"] = np.array(5)
            save_dict[f"ep{i}_context_length"] = np.array(2)
        npz_path = tmp_path / "imagination_data.npz"
        np.savez_compressed(npz_path, **save_dict)
        episodes = load_imagination_data(str(npz_path))
        assert len(episodes) == 2
        assert "real_frames" in episodes[0]


class TestComputeCheckpointMetrics:
    """Tests for compute_checkpoint_metrics end-to-end."""

    def test_creates_metrics_json(self, tmp_path):
        """Create a fake imagination NPZ and verify metrics.json is produced."""
        from snake_env.snake_env import SnakeEnv
        env = SnakeEnv()
        obs, _ = env.reset(seed=42)

        save_dict = {"num_episodes": np.array(1)}
        frames = np.stack([obs, obs, obs])
        save_dict["ep0_real_frames"] = frames
        save_dict["ep0_imagined_frames"] = frames
        save_dict["ep0_context_recon"] = frames[:2]
        save_dict["ep0_context_real"] = frames[:2]
        save_dict["ep0_actions"] = np.zeros((3, 4), dtype=np.float32)
        save_dict["ep0_rewards"] = np.zeros(3, dtype=np.float32)
        save_dict["ep0_is_terminal"] = np.zeros(3, dtype=bool)
        save_dict["ep0_episode_length"] = np.array(5)
        save_dict["ep0_context_length"] = np.array(2)

        npz_path = tmp_path / "imagination_data.npz"
        np.savez_compressed(npz_path, **save_dict)

        metrics = compute_checkpoint_metrics(str(npz_path))
        assert "overall" in metrics
        assert "per_episode" in metrics

        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()

    def test_skips_existing_metrics(self, tmp_path):
        """If metrics.json already exists, it should be loaded not recomputed."""
        existing = {"overall": {"mse": {"mean": 0.5}}}
        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing, f)

        npz_path = tmp_path / "imagination_data.npz"
        np.savez_compressed(npz_path, num_episodes=np.array(0))

        result = compute_checkpoint_metrics(str(npz_path))
        assert result["overall"]["mse"]["mean"] == 0.5
