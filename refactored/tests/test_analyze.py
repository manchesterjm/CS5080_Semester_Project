"""Tests for analyze module."""

import json
import pathlib
import tempfile

import numpy as np
import pytest

from analyze import (
    load_all_checkpoint_metrics,
    load_eval_performance,
    match_performance_to_checkpoints,
    generate_summary_table,
    _compute_h1_trends,
    _is_improving,
    _compute_early_late_ttest,
    _compute_h2_correlations,
    _compute_h3_correlations,
    _print_h3_correlations,
    _compute_fisher_z,
    run_analysis,
    BETTER_DIRECTION,
)
from analyze import test_h1 as analyze_test_h1
from analyze import test_h2 as analyze_test_h2
from analyze import test_h3 as analyze_test_h3
from constants import METRIC_NAMES


def _make_matched_data(n_checkpoints=10):
    """Create synthetic matched_data for testing hypothesis functions."""
    matched = []
    for i in range(n_checkpoints):
        step = (i + 1) * 50000
        # Metrics that improve over training
        frac = i / max(n_checkpoints - 1, 1)
        metrics = {
            "overall": {
                "mse": {"mean": 0.5 - 0.3 * frac, "std": 0.1},
                "ssim": {"mean": 0.5 + 0.4 * frac, "std": 0.1},
                "head_error": {"mean": 5.0 - 3.0 * frac, "std": 1.0},
                "body_accuracy": {"mean": 0.3 + 0.5 * frac, "std": 0.1},
                "food_correct": {"mean": 0.2 + 0.6 * frac, "std": 0.1},
            },
            "per_step": [
                {
                    "step": t,
                    "mse_mean": 0.1 + 0.01 * t,
                    "ssim_mean": 0.9 - 0.01 * t,
                    "head_error_mean": 1.0 + 0.1 * t,
                    "body_accuracy_mean": 0.8 - 0.01 * t,
                    "food_correct_mean": 0.7 - 0.01 * t,
                }
                for t in range(10)
            ],
        }
        eval_return = -2.0 + 10.0 * frac
        matched.append((step, metrics, eval_return, 100 + i * 10))
    return matched


class TestIsImproving:
    def test_lower_is_better_negative_rho(self):
        assert _is_improving("lower", -0.5) is True

    def test_lower_is_better_positive_rho(self):
        assert _is_improving("lower", 0.5) is False

    def test_higher_is_better_positive_rho(self):
        assert _is_improving("higher", 0.5) is True


class TestBetterDirection:
    def test_all_metrics_covered(self):
        for name in METRIC_NAMES:
            assert name in BETTER_DIRECTION


class TestLoadCheckpointMetrics:
    def test_loads_sorted_by_step(self, tmp_path):
        for step in [200000, 100000, 300000]:
            ckpt_dir = tmp_path / f"checkpoint_{step:07d}"
            ckpt_dir.mkdir()
            with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"step": step}, f)
        results = load_all_checkpoint_metrics(str(tmp_path))
        assert len(results) == 3
        assert results[0][0] == 100000
        assert results[2][0] == 300000

    def test_empty_dir_returns_empty(self, tmp_path):
        assert load_all_checkpoint_metrics(str(tmp_path)) == []


class TestLoadEvalPerformance:
    def test_parses_eval_entries(self, tmp_path):
        jsonl = tmp_path / "metrics.jsonl"
        lines = [
            json.dumps({"step": 1000, "eval_return": 5.0, "eval_length": 100}),
            json.dumps({"step": 2000, "train_loss": 0.1}),
            json.dumps({"step": 3000, "eval_return": 8.0}),
        ]
        jsonl.write_text("\n".join(lines))
        perf = load_eval_performance(str(jsonl))
        assert len(perf) == 2
        assert perf[1000]["eval_return"] == 5.0
        assert perf[3000]["eval_length"] is None


class TestMatchPerformance:
    def test_matches_close_steps(self):
        ckpt_metrics = [(100000, {"data": 1}), (200000, {"data": 2})]
        eval_perf = {
            99000: {"eval_return": 5.0, "eval_length": 100},
            201000: {"eval_return": 8.0, "eval_length": 200},
        }
        matched = match_performance_to_checkpoints(ckpt_metrics, eval_perf)
        assert len(matched) == 2
        assert matched[0][2] == 5.0

    def test_skips_distant_steps(self):
        ckpt_metrics = [(100000, {"data": 1})]
        eval_perf = {500000: {"eval_return": 5.0}}
        matched = match_performance_to_checkpoints(ckpt_metrics, eval_perf)
        assert len(matched) == 0


class TestComputeH1Trends:
    def test_returns_results_for_all_metrics(self):
        matched = _make_matched_data(10)
        results = _compute_h1_trends(matched)
        for name in METRIC_NAMES:
            assert name in results
            assert "spearman_rho" in results[name]
            assert "p_value" in results[name]
            assert "significant" in results[name]
            assert "improving" in results[name]


class TestComputeEarlyLateTtest:
    def test_returns_none_for_small_data(self):
        matched = _make_matched_data(4)
        result = _compute_early_late_ttest(matched)
        assert result is None

    def test_returns_dict_for_sufficient_data(self):
        matched = _make_matched_data(10)
        result = _compute_early_late_ttest(matched)
        assert result is not None
        assert "k" in result
        assert "metrics" in result


class TestComputeH2Correlations:
    def test_returns_all_metrics(self):
        matched = _make_matched_data(10)
        results = _compute_h2_correlations(matched)
        for name in METRIC_NAMES:
            assert name in results
            assert "spearman_rho" in results[name]
            assert "pearson_r" in results[name]


class TestComputeH3:
    def test_correlations_computed(self):
        matched = _make_matched_data(10)
        correlations = _compute_h3_correlations(matched)
        assert "mse" in correlations
        assert "head_error" in correlations

    def test_print_returns_averages(self):
        matched = _make_matched_data(10)
        correlations = _compute_h3_correlations(matched)
        avg_pixel, avg_semantic = _print_h3_correlations(correlations)
        assert isinstance(avg_pixel, (float, np.floating))
        assert isinstance(avg_semantic, (float, np.floating))

    def test_fisher_z_with_enough_data(self):
        matched = _make_matched_data(10)
        correlations = _compute_h3_correlations(matched)
        result = _compute_fisher_z(correlations, 10)
        assert result is not None
        assert "z_statistic" in result
        assert "p_value" in result

    def test_fisher_z_too_few_points(self):
        matched = _make_matched_data(3)
        correlations = _compute_h3_correlations(matched)
        result = _compute_fisher_z(correlations, 3)
        assert result is None


class TestHypothesisFunctions:
    """Integration tests for test_h1, test_h2, test_h3 with figure output."""

    def test_h1_produces_figures(self, tmp_path):
        matched = _make_matched_data(10)
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        results = analyze_test_h1(matched, figures_dir)
        assert (figures_dir / "h1_imagination_vs_training.png").exists()
        assert (figures_dir / "h1_horizon_degradation.png").exists()
        assert "mse" in results

    def test_h2_produces_figures(self, tmp_path):
        matched = _make_matched_data(10)
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        results = analyze_test_h2(matched, figures_dir)
        assert (figures_dir / "h2_imagination_vs_performance.png").exists()

    def test_h3_produces_figures(self, tmp_path):
        matched = _make_matched_data(10)
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        results = analyze_test_h3(matched, figures_dir)
        assert (figures_dir / "h3_pixel_vs_semantic.png").exists()
        assert "correlations" in results
        assert "fisher_z" in results


class TestGenerateSummaryTable:
    def test_creates_csv(self, tmp_path):
        matched = _make_matched_data(3)
        csv_path = tmp_path / "summary.csv"
        generate_summary_table(matched, csv_path)
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows


class TestRunAnalysis:
    def test_full_pipeline(self, tmp_path):
        """End-to-end test with synthetic checkpoint data."""
        # Create fake checkpoint metrics
        matched = _make_matched_data(5)
        for step, metrics, _, _ in matched:
            ckpt_dir = tmp_path / f"checkpoint_{step:07d}"
            ckpt_dir.mkdir()
            with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f)

        # Create fake metrics.jsonl
        jsonl_path = tmp_path / "metrics.jsonl"
        lines = []
        for step, _, eval_ret, eval_len in matched:
            lines.append(json.dumps({
                "step": step, "eval_return": eval_ret, "eval_length": eval_len,
            }))
        jsonl_path.write_text("\n".join(lines))

        results = run_analysis(str(tmp_path), str(jsonl_path))
        assert results is not None
        assert "h1_imagination_improves" in results
        assert "h2_correlates_with_performance" in results
        assert "h3_semantic_vs_pixel" in results
        assert (tmp_path / "figures").is_dir()
        assert (tmp_path / "hypothesis_results.json").exists()

    def test_returns_none_with_no_data(self, tmp_path):
        jsonl_path = tmp_path / "metrics.jsonl"
        jsonl_path.write_text("")
        result = run_analysis(str(tmp_path), str(jsonl_path))
        assert result is None
