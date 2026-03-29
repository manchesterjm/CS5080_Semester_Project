"""Tests for run_analysis module."""

import pathlib

import pytest

from run_analysis import get_paths, discover_checkpoints


class TestGetPaths:
    def test_returns_expected_keys(self):
        paths = get_paths("snake_1M")
        assert "checkpoint_dir" in paths
        assert "eval_dir" in paths
        assert "metrics_jsonl" in paths
        assert "analysis_dir" in paths

    def test_paths_include_run_name(self):
        paths = get_paths("my_run")
        assert "my_run" in str(paths["checkpoint_dir"])
        assert "my_run" in str(paths["analysis_dir"])

    def test_returns_pathlib_paths(self):
        paths = get_paths("test")
        for key in paths:
            assert isinstance(paths[key], pathlib.Path)


class TestDiscoverCheckpoints:
    def test_all_glob(self, tmp_path):
        for step in [50000, 100000, 150000]:
            (tmp_path / f"checkpoint_{step:07d}.pt").touch()
        paths = discover_checkpoints("all", tmp_path)
        assert len(paths) == 3
        assert paths[0].name == "checkpoint_0050000.pt"

    def test_specific_steps(self, tmp_path):
        (tmp_path / "checkpoint_0100000.pt").touch()
        (tmp_path / "checkpoint_0200000.pt").touch()
        paths = discover_checkpoints("100000,200000", tmp_path)
        assert len(paths) == 2

    def test_missing_checkpoint_warns(self, tmp_path, capsys):
        paths = discover_checkpoints("999999", tmp_path)
        assert len(paths) == 0
        assert "WARNING" in capsys.readouterr().out

    def test_empty_dir(self, tmp_path):
        paths = discover_checkpoints("all", tmp_path)
        assert len(paths) == 0
