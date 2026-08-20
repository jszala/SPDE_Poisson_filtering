from __future__ import annotations

from dataclasses import replace

import numpy as np

from spde_poisson_filtering.filter import BootstrapParticleFilter
from spde_poisson_filtering.observations import simulate_observations
from spde_poisson_filtering.spde import simulate_signal


def test_filter_outputs_are_valid(tiny_config) -> None:
    truth = simulate_signal(tiny_config)
    observations = simulate_observations(truth, tiny_config)
    result = BootstrapParticleFilter(tiny_config, 4).run(observations.counts[4], truth)
    assert result.mean_u.shape == truth.activator.shape
    assert np.all(np.isfinite(result.mean_u))
    assert np.all((result.ess >= 1) & (result.ess <= tiny_config.filter.particles + 1e-10))
    assert result.resampling_steps.ndim == 1


def test_serial_and_two_worker_results_agree(tiny_config) -> None:
    truth = simulate_signal(tiny_config)
    observations = simulate_observations(truth, tiny_config)
    serial = BootstrapParticleFilter(tiny_config, 4, workers=1).run(observations.counts[4], truth)
    parallel_config = replace(tiny_config, execution=replace(tiny_config.execution, workers=2))
    parallel = BootstrapParticleFilter(parallel_config, 4, workers=2).run(
        observations.counts[4], truth
    )
    np.testing.assert_allclose(serial.mean_u, parallel.mean_u, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(serial.mean_v, parallel.mean_v, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(serial.ess, parallel.ess, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        serial.log_normalizers, parallel.log_normalizers, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_array_equal(serial.resampled, parallel.resampled)
