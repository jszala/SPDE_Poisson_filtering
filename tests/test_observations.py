from __future__ import annotations

import numpy as np

from spde_poisson_filtering.observations import (
    aggregate_blocks,
    intensity_density,
    simulate_observations,
)
from spde_poisson_filtering.spde import simulate_signal


def test_block_aggregation_conserves_mass() -> None:
    values = np.arange(2 * 8 * 8).reshape(2, 8, 8)
    reduced = aggregate_blocks(values, 2)
    assert reduced.shape == (2, 2, 2)
    np.testing.assert_array_equal(reduced.sum(axis=(-2, -1)), values.sum(axis=(-2, -1)))


def test_intensity_is_positive_bounded_and_uses_nonnegative_state(tiny_config) -> None:
    state = np.array([[-2.0, 0.0], [0.5, 1000.0]])
    intensity = intensity_density(state, 0.0, tiny_config.observation)
    assert intensity.min() >= tiny_config.observation.lambda_min
    assert intensity.max() <= tiny_config.observation.lambda_max
    assert intensity[0, 0] == intensity[0, 1]


def test_observation_resolutions_share_and_conserve_events(tiny_config) -> None:
    signal = simulate_signal(tiny_config)
    observations = simulate_observations(signal, tiny_config)
    fine = observations.counts[4]
    coarse = observations.counts[2]
    assert np.issubdtype(fine.dtype, np.integer)
    assert np.all(fine >= 0)
    np.testing.assert_array_equal(fine.sum(axis=(-2, -1)), coarse.sum(axis=(-2, -1)))
