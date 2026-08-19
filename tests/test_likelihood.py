from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from spde_poisson_filtering.likelihood import (
    normalize_log_weights,
    poisson_log_likelihood,
    poisson_log_weight_increment,
)


def test_full_likelihood_matches_scipy() -> None:
    counts = np.array([0, 2, 7])
    rates = np.array([0.2, 1.5, 4.0])
    dt = 0.25
    actual = poisson_log_likelihood(counts, rates, dt)
    expected = poisson.logpmf(counts, dt * rates).sum()
    assert actual == pytest.approx(expected)


def test_reference_form_differs_by_particle_independent_constant() -> None:
    counts = np.array([2, 0, 1])
    rates = np.array([[0.2, 1.1, 2.0], [3.0, 0.5, 0.7]])
    full = poisson_log_likelihood(counts, rates, 0.1, axis=1)
    relative = poisson_log_weight_increment(counts, rates, 0.1, reference_intensity=0.8, axis=1)
    difference = full - relative
    assert difference[0] == pytest.approx(difference[1])


@pytest.mark.parametrize(
    ("counts", "rates"),
    [([-1], [1.0]), ([0.5], [1.0]), ([0], [0.0]), ([0], [np.inf])],
)
def test_invalid_inputs_fail(counts: list[float], rates: list[float]) -> None:
    with pytest.raises(ValueError):
        poisson_log_likelihood(counts, rates, 1.0)


def test_log_weight_normalization_handles_large_values() -> None:
    weights, normalizer = normalize_log_weights([10_000.0, 9_999.0])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1]
    assert np.isfinite(normalizer)


def test_zero_count_and_extreme_positive_rate_remain_finite() -> None:
    value = poisson_log_likelihood([0, 0], [1.0e-12, 1.0e12], 1.0e-6)
    assert np.isfinite(value)
