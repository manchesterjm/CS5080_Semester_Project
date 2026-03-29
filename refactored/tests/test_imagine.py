"""Tests for imagine module (non-GPU components only)."""

import pathlib

import numpy as np
import pytest

from imagine import load_eval_episodes, _build_episode_tensors


class TestLoadEvalEpisodes:
    def test_loads_npz_files(self, tmp_path):
        for i in range(3):
            ep = {
                "image": np.zeros((20, 64, 64, 3), dtype=np.uint8),
                "action": np.zeros((20, 4), dtype=np.float32),
                "reward": np.zeros(20, dtype=np.float32),
                "is_first": np.zeros(20, dtype=bool),
                "is_terminal": np.zeros(20, dtype=bool),
            }
            np.savez(tmp_path / f"ep_{i:03d}.npz", **ep)
        episodes = load_eval_episodes(str(tmp_path))
        assert len(episodes) == 3
        assert "image" in episodes[0]

    def test_filters_short_episodes(self, tmp_path):
        short_ep = {
            "image": np.zeros((3, 64, 64, 3), dtype=np.uint8),
            "action": np.zeros((3, 4), dtype=np.float32),
            "reward": np.zeros(3, dtype=np.float32),
            "is_first": np.zeros(3, dtype=bool),
            "is_terminal": np.zeros(3, dtype=bool),
        }
        np.savez(tmp_path / "ep_short.npz", **short_ep)
        episodes = load_eval_episodes(str(tmp_path), min_length=10)
        assert len(episodes) == 0

    def test_max_episodes_cap(self, tmp_path):
        for i in range(5):
            ep = {
                "image": np.zeros((20, 64, 64, 3), dtype=np.uint8),
                "action": np.zeros((20, 4), dtype=np.float32),
                "reward": np.zeros(20, dtype=np.float32),
                "is_first": np.zeros(20, dtype=bool),
                "is_terminal": np.zeros(20, dtype=bool),
            }
            np.savez(tmp_path / f"ep_{i:03d}.npz", **ep)
        episodes = load_eval_episodes(str(tmp_path), max_episodes=2)
        assert len(episodes) == 2

    def test_empty_dir(self, tmp_path):
        episodes = load_eval_episodes(str(tmp_path))
        assert len(episodes) == 0


class TestBuildEpisodeTensors:
    def test_creates_correct_keys(self):
        import torch
        episode = {
            "image": np.zeros((5, 64, 64, 3), dtype=np.uint8),
            "action": np.zeros((5, 4), dtype=np.float32),
            "is_first": np.zeros(5, dtype=bool),
            "is_terminal": np.zeros(5, dtype=bool),
            "reward": np.zeros(5, dtype=np.float32),
        }
        result = _build_episode_tensors(episode, "cpu")
        assert set(result.keys()) == {"image", "action", "is_first", "is_terminal", "reward"}
        for v in result.values():
            assert isinstance(v, torch.Tensor)
            assert v.shape[0] == 1  # batch dimension
