from __future__ import annotations

import numpy as np
import pytest

from spde_poisson_filtering.resampling import effective_sample_size, residual_resample


def test_effective_sample_size_extremes() -> None:
    assert effective_sample_size([0.25] * 4) == pytest.approx(4.0)
    assert effective_sample_size([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_residual_resampling_keeps_deterministic_copies() -> None:
    rng = np.random.default_rng(7)
    indices = residual_resample([0.6, 0.2, 0.1, 0.1], rng)
    assert indices.shape == (4,)
    assert np.count_nonzero(indices == 0) >= 2
    assert np.all((0 <= indices) & (indices < 4))


def test_residual_resampling_is_seeded() -> None:
    one = residual_resample([0.4, 0.3, 0.2, 0.1], np.random.default_rng(12))
    two = residual_resample([0.4, 0.3, 0.2, 0.1], np.random.default_rng(12))
    np.testing.assert_array_equal(one, two)
