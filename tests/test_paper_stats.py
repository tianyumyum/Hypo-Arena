"""Sanity tests for scripts/paper/_stats.py rank correlations."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from scripts._paper_stats import (
    average_ranks,
    fraction_at_least,
    kendall_tau,
    mean_std,
    spearman_rho,
)


def test_average_ranks_no_ties():
    assert average_ranks([10.0, 30.0, 20.0]) == [1.0, 3.0, 2.0]


def test_average_ranks_with_ties():
    # values [4, 4, 1, 8] → sorted ranks 2.5, 2.5, 1, 4
    assert average_ranks([4.0, 4.0, 1.0, 8.0]) == [2.5, 2.5, 1.0, 4.0]


def test_spearman_perfect_positive_is_one():
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative_is_minus_one():
    assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    # both lists tied identically → perfect monotone via midranks
    assert spearman_rho([1.0, 1.0, 2.0, 3.0], [10, 10, 20, 30]) == pytest.approx(1.0)


def test_kendall_perfect_positive_is_one():
    assert kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_kendall_perfect_negative_is_minus_one():
    assert kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_kendall_zero_for_random_arrangement():
    # 1 concordant, 1 discordant → tau = 0
    val = kendall_tau([1, 2, 3], [2, 1, 3])
    assert -1e-9 <= val <= 1e-9 or val == pytest.approx(1.0 / 3)
    # exact: pairs (1,2)/(2,1)→discordant, (1,3)/(2,3)→concordant, (2,3)/(1,3)→concordant ⇒ 2 conc 1 disc / 3 = 1/3
    assert val == pytest.approx(1.0 / 3)


def test_fraction_at_least():
    assert fraction_at_least([3.0, 4.0, 5.0, 4.5], 4.0) == pytest.approx(0.75)
    assert fraction_at_least([], 1.0) == 0.0


def test_mean_std_small_sample():
    mean, std = mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert mean == pytest.approx(5.0)
    # Sample std = sqrt( sum((x-5)^2) / (n-1) ) = sqrt(32/7)
    assert std == pytest.approx(math.sqrt(32 / 7))


def test_mean_std_singleton():
    assert mean_std([3.0]) == (3.0, 0.0)


def test_mean_std_empty():
    assert mean_std([]) == (0.0, 0.0)
