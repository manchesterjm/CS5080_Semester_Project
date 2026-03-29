"""Tests for constants module."""

from constants import (
    REWARD_FOOD, REWARD_COLLISION, STEP_PENALTY, INITIAL_SNAKE_LENGTH,
    NUM_ACTIONS, IMAGE_SIZE, CELL_SIZE, PADDING, METRIC_NAMES,
    PIXEL_METRICS, SEMANTIC_METRICS, FIGURE_DPI, P_VALUE_THRESHOLD,
    DISPLAY_SCALE, DISPLAY_SIZE, DEFAULT_FPS, MAX_PLAY_STEPS,
)


def test_reward_values():
    assert REWARD_FOOD == 1.0
    assert REWARD_COLLISION == -1.0
    assert STEP_PENALTY == -0.002


def test_grid_constants():
    assert INITIAL_SNAKE_LENGTH == 3
    assert NUM_ACTIONS == 4
    assert IMAGE_SIZE == 64
    assert CELL_SIZE == 6
    assert PADDING == 2


def test_display_size_derived():
    assert DISPLAY_SIZE == IMAGE_SIZE * DISPLAY_SCALE
    assert DISPLAY_SCALE == 8


def test_metric_names_tuple():
    assert isinstance(METRIC_NAMES, tuple)
    assert len(METRIC_NAMES) == 5
    assert "mse" in METRIC_NAMES
    assert "ssim" in METRIC_NAMES


def test_pixel_semantic_partition():
    all_metrics = set(PIXEL_METRICS) | set(SEMANTIC_METRICS)
    assert all_metrics == set(METRIC_NAMES)
    assert set(PIXEL_METRICS) & set(SEMANTIC_METRICS) == set()


def test_analysis_thresholds():
    assert FIGURE_DPI == 150
    assert P_VALUE_THRESHOLD == 0.05


def test_playback_defaults():
    assert DEFAULT_FPS == 10
    assert MAX_PLAY_STEPS == 100_000
