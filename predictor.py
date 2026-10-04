"""
predictor.py — Monte Carlo simulation engine for the century predictor.

This module contains pure Python/NumPy logic with no Streamlit dependency,
making it independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioParams:
    """Parameters for a single simulation scenario."""
    name: str
    century_prob: float          # probability of a century in any given innings
    label: str = ""              # short display label


@dataclass
class SimulationResult:
    """Aggregated results from a Monte Carlo run."""
    scenario_name: str
    prob_reach_target: float     # 0–1 probability of reaching or exceeding target
    prob_miss_target: float      # 1 - prob_reach_target
    expected_additional: float   # mean additional centuries across simulations
    expected_final_total: float  # mean final total
    median_final_total: float
    p10_final_total: float       # 10th percentile
    p90_final_total: float       # 90th percentile
    final_totals: np.ndarray     # raw per-simulation final totals (for histogram)
    reached: np.ndarray          # bool array: did each sim reach target?

    # Human-readable summary
    @property
    def reaches_target(self) -> bool:
        return self.prob_reach_target >= 0.5


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(
    current_centuries: int,
    target: int,
    remaining_innings: int,
    century_prob: float,
    n_simulations: int = 10_000,
    seed: int = 42,
    scenario_name: str = "Base Case",
) -> SimulationResult:
    """
    Monte Carlo simulation: for each of `n_simulations` runs, simulate
    `remaining_innings` Bernoulli trials with success probability `century_prob`.
    Record whether the cumulative total reaches `target`.

    Parameters
    ----------
    current_centuries : int   Current confirmed century count.
    target            : int   Target century count (default 100).
    remaining_innings : int   Future innings to simulate.
    century_prob      : float Probability of a century per innings (0 < p ≤ 1).
    n_simulations     : int   Number of Monte Carlo repetitions.
    seed              : int   Random seed for reproducibility.
    scenario_name     : str   Label for the scenario.

    Returns
    -------
    SimulationResult dataclass with all aggregated statistics.
    """
    if not (0.0 < century_prob <= 1.0):
        raise ValueError(f"century_prob must be in (0, 1]; got {century_prob}")
    if remaining_innings < 0:
        raise ValueError("remaining_innings must be >= 0")
    if current_centuries < 0:
        raise ValueError("current_centuries must be >= 0")
    if target < 0:
        raise ValueError("target must be >= 0")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")

    rng = np.random.default_rng(seed)

    if remaining_innings == 0:
        # No future play — result is deterministic
        final_total = current_centuries
        reached_target = bool(final_total >= target)
        final_totals = np.full(n_simulations, final_total, dtype=np.int32)
        reached = np.full(n_simulations, reached_target, dtype=bool)
    else:
        # Shape: (n_simulations, remaining_innings) Bernoulli matrix
        innings_matrix = rng.random(size=(n_simulations, remaining_innings)) < century_prob
        additional = innings_matrix.sum(axis=1).astype(np.int32)
        final_totals = (current_centuries + additional).astype(np.int32)
        reached = final_totals >= target

    additional_centuries = final_totals - current_centuries

    return SimulationResult(
        scenario_name=scenario_name,
        prob_reach_target=float(reached.mean()),
        prob_miss_target=float((~reached).mean()),
        expected_additional=float(additional_centuries.mean()),
        expected_final_total=float(final_totals.mean()),
        median_final_total=float(np.median(final_totals)),
        p10_final_total=float(np.percentile(final_totals, 10)),
        p90_final_total=float(np.percentile(final_totals, 90)),
        final_totals=final_totals,
        reached=reached,
    )


# ---------------------------------------------------------------------------
# Historical data helpers
# ---------------------------------------------------------------------------

def load_innings_data(csv_path: str | Path) -> Optional[pd.DataFrame]:
    """
    Load the innings CSV. Returns None if the file does not exist or is empty.
    Only includes rows where format is one of: Test, ODI, T20I.
    """
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    required = {"format", "century"}
    if not required.issubset(df.columns):
        return None

    international_formats = {"Test", "ODI", "T20I"}
    df = df[df["format"].isin(international_formats)].copy()
    if df.empty:
        return None
    return df


def estimate_century_prob_from_data(df: pd.DataFrame) -> tuple[float, dict]:
    """
    Estimate the historical century probability per innings from a loaded
    innings DataFrame.

    Returns
    -------
    (probability, metadata_dict) where metadata explains the calculation.
    """
    total_innings = len(df)
    century_innings = int(df["century"].sum())
    probability = century_innings / total_innings if total_innings > 0 else 0.0

    meta = {
        "total_innings": total_innings,
        "century_innings": century_innings,
        "formats": df["format"].value_counts().to_dict(),
    }
    return round(probability, 4), meta
