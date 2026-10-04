"""
tests/test_predictor.py

Basic test suite for the Monte Carlo prediction engine.
Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

# Add parent directory to sys.path so predictor can be imported
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from predictor import (
    estimate_century_prob_from_data,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(total_innings: int, centuries: int, fmt: str = "ODI") -> pd.DataFrame:
    """Build a minimal innings DataFrame for testing."""
    rows = (
        [{"format": fmt, "century": 1}] * centuries
        + [{"format": fmt, "century": 0}] * (total_innings - centuries)
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Probability boundary tests
# ---------------------------------------------------------------------------

class TestProbabilityBoundaries:
    def test_invalid_prob_zero_raises(self):
        with pytest.raises(ValueError, match="century_prob"):
            run_simulation(86, 100, 50, century_prob=0.0)

    def test_invalid_prob_negative_raises(self):
        with pytest.raises(ValueError, match="century_prob"):
            run_simulation(86, 100, 50, century_prob=-0.1)

    def test_invalid_prob_above_one_raises(self):
        with pytest.raises(ValueError, match="century_prob"):
            run_simulation(86, 100, 50, century_prob=1.01)

    def test_prob_exactly_one_all_succeed(self):
        """With p=1.0 every innings is a century; all sims must reach target."""
        result = run_simulation(
            current_centuries=86, target=100, remaining_innings=50,
            century_prob=1.0, n_simulations=1_000, seed=42,
        )
        assert result.prob_reach_target == 1.0

    def test_prob_very_small_almost_none_succeed(self):
        """With p=0.001 and 5 innings left, reaching the target is near impossible."""
        result = run_simulation(
            current_centuries=86, target=100, remaining_innings=5,
            century_prob=0.001, n_simulations=10_000, seed=42,
        )
        assert result.prob_reach_target < 0.01


# ---------------------------------------------------------------------------
# Zero remaining innings
# ---------------------------------------------------------------------------

class TestZeroRemainingInnings:
    def test_target_not_reached(self):
        result = run_simulation(
            current_centuries=86, target=100, remaining_innings=0,
            century_prob=0.15, n_simulations=1_000, seed=42,
        )
        assert result.prob_reach_target == 0.0
        assert result.expected_additional == 0.0
        assert result.expected_final_total == 86.0

    def test_target_already_met_with_zero_innings(self):
        result = run_simulation(
            current_centuries=100, target=100, remaining_innings=0,
            century_prob=0.15, n_simulations=1_000, seed=42,
        )
        assert result.prob_reach_target == 1.0


# ---------------------------------------------------------------------------
# Already reached target
# ---------------------------------------------------------------------------

class TestAlreadyReachedTarget:
    def test_already_at_target(self):
        result = run_simulation(
            current_centuries=100, target=100, remaining_innings=50,
            century_prob=0.15, n_simulations=1_000, seed=42,
        )
        assert result.prob_reach_target == 1.0

    def test_already_exceeds_target(self):
        result = run_simulation(
            current_centuries=110, target=100, remaining_innings=50,
            century_prob=0.15, n_simulations=1_000, seed=42,
        )
        assert result.prob_reach_target == 1.0


# ---------------------------------------------------------------------------
# Correct target gap
# ---------------------------------------------------------------------------

class TestTargetGap:
    def test_gap_reflected_in_expected_additional(self):
        """Expected additional should be close to prob * innings (law of large numbers)."""
        prob = 0.15
        innings = 100
        result = run_simulation(
            current_centuries=86, target=100, remaining_innings=innings,
            century_prob=prob, n_simulations=100_000, seed=42,
        )
        expected = prob * innings
        # Allow ±5 % relative tolerance
        assert abs(result.expected_additional - expected) / expected < 0.05

    def test_final_total_equals_current_plus_additional(self):
        current = 86
        result = run_simulation(
            current_centuries=current, target=100, remaining_innings=50,
            century_prob=0.15, n_simulations=1_000, seed=42,
        )
        np.testing.assert_array_equal(
            result.final_totals,
            current + (result.final_totals - current),
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        kwargs = dict(
            current_centuries=86, target=100, remaining_innings=100,
            century_prob=0.15, n_simulations=10_000, seed=42,
        )
        r1 = run_simulation(**kwargs)
        r2 = run_simulation(**kwargs)
        assert r1.prob_reach_target == r2.prob_reach_target
        np.testing.assert_array_equal(r1.final_totals, r2.final_totals)

    def test_different_seeds_differ(self):
        base_kwargs = dict(
            current_centuries=86, target=100, remaining_innings=100,
            century_prob=0.15, n_simulations=10_000,
        )
        r1 = run_simulation(**base_kwargs, seed=1)
        r2 = run_simulation(**base_kwargs, seed=2)
        # Results should not be identical (probability of collision is negligible)
        assert not np.array_equal(r1.final_totals, r2.final_totals)


# ---------------------------------------------------------------------------
# Output range validation
# ---------------------------------------------------------------------------

class TestOutputRanges:
    def _run(self, **kwargs):
        defaults = dict(
            current_centuries=86, target=100, remaining_innings=100,
            century_prob=0.15, n_simulations=5_000, seed=42,
        )
        defaults.update(kwargs)
        return run_simulation(**defaults)

    def test_prob_in_unit_interval(self):
        r = self._run()
        assert 0.0 <= r.prob_reach_target <= 1.0
        assert 0.0 <= r.prob_miss_target <= 1.0

    def test_probs_sum_to_one(self):
        r = self._run()
        assert abs(r.prob_reach_target + r.prob_miss_target - 1.0) < 1e-9

    def test_final_totals_shape(self):
        n = 5_000
        r = self._run(n_simulations=n)
        assert r.final_totals.shape == (n,)

    def test_final_totals_lower_bounded(self):
        current = 86
        r = self._run(current_centuries=current)
        assert int(r.final_totals.min()) >= current

    def test_percentiles_ordered(self):
        r = self._run()
        assert r.p10_final_total <= r.median_final_total <= r.p90_final_total

    def test_expected_additional_non_negative(self):
        r = self._run()
        assert r.expected_additional >= 0.0


# ---------------------------------------------------------------------------
# Historical data probability estimation
# ---------------------------------------------------------------------------

class TestEstimateCenturyProb:
    def test_basic_estimate(self):
        df = _make_df(total_innings=100, centuries=15)
        prob, meta = estimate_century_prob_from_data(df)
        assert prob == pytest.approx(0.15)
        assert meta["total_innings"] == 100
        assert meta["century_innings"] == 15

    def test_zero_centuries(self):
        df = _make_df(total_innings=50, centuries=0)
        prob, meta = estimate_century_prob_from_data(df)
        assert prob == 0.0

    def test_all_centuries(self):
        df = _make_df(total_innings=10, centuries=10)
        prob, meta = estimate_century_prob_from_data(df)
        assert prob == pytest.approx(1.0)

    def test_non_international_formats_excluded(self):
        """IPL rows should be excluded before this function is called (by load_innings_data)."""
        # Simulate a pre-filtered DataFrame with only international formats
        df = pd.DataFrame([
            {"format": "Test", "century": 1},
            {"format": "ODI",  "century": 0},
            {"format": "T20I", "century": 0},
        ])
        prob, meta = estimate_century_prob_from_data(df)
        # estimate_century_prob_from_data rounds to 4 decimal places; use rel tolerance
        assert prob == pytest.approx(1 / 3, rel=1e-3)
        assert meta["total_innings"] == 3
