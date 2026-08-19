from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from spde_poisson_filtering.spde import FHNStepper, neumann_laplacian_2d, simulate_signal


def test_neumann_laplacian_annihilates_constants() -> None:
    operator = neumann_laplacian_2d(5, 4, 0.2)
    np.testing.assert_allclose(operator @ np.ones(20), 0.0, atol=1e-13)
    np.testing.assert_allclose(operator.toarray(), operator.toarray().T)


def test_neumann_laplacian_respects_array_axis_order() -> None:
    dx = 0.2
    field = np.arange(12, dtype=np.float64).reshape(3, 4)
    actual = (neumann_laplacian_2d(3, 4, dx) @ field.ravel()).reshape(field.shape)
    expected = np.zeros_like(field)
    expected[1:] += field[:-1] - field[1:]
    expected[:-1] += field[1:] - field[:-1]
    expected[:, 1:] += field[:, :-1] - field[:, 1:]
    expected[:, :-1] += field[:, 1:] - field[:, :-1]
    np.testing.assert_allclose(actual, expected / dx**2, atol=1e-12)


def test_constant_diffusion_state_is_unchanged(tiny_config) -> None:
    model = replace(
        tiny_config.model,
        skew=0.0,
        potential=0.0,
        gamma=0.0,
        sigma_u=0.0,
        sigma_v=0.0,
    )
    stepper = FHNStepper(tiny_config.grid, model)
    u = np.full((1, 4, 4), 0.7)
    v = np.zeros_like(u)
    next_u, next_v = stepper.step(u, v, seeds=None)
    np.testing.assert_allclose(next_u, u, atol=1e-13)
    np.testing.assert_allclose(next_v, v, atol=1e-13)


def test_white_noise_has_euler_maruyama_scaling(tiny_config) -> None:
    model = replace(
        tiny_config.model,
        diffusion_u=0.0,
        diffusion_v=0.0,
        skew=0.0,
        potential=0.0,
        gamma=0.0,
        sigma_u=0.3,
        sigma_v=0.0,
    )
    stepper = FHNStepper(tiny_config.grid, model)
    u = np.zeros((1, 4, 4))
    v = np.zeros_like(u)
    seed = np.array([42], dtype=np.uint64)
    next_u, _ = stepper.step(u, v, seed)
    expected = (
        model.sigma_u
        * np.sqrt(tiny_config.grid.dt)
        / tiny_config.grid.dx
        * np.random.default_rng(42).standard_normal((4, 4))
    )
    np.testing.assert_allclose(next_u[0], expected)


def test_step_matches_direct_sparse_reference(tiny_config) -> None:
    model = replace(tiny_config.model, sigma_u=0.0, sigma_v=0.0)
    grid = tiny_config.grid
    rng = np.random.default_rng(4)
    u = rng.uniform(0.4, 0.9, size=(1, grid.nx, grid.ny))
    v = rng.uniform(0.0, 0.2, size=u.shape)

    actual_u, actual_v = FHNStepper(grid, model).step(u, v, seeds=None)
    laplacian = neumann_laplacian_2d(grid.nx, grid.ny, grid.dx)
    identity = eye(grid.nx * grid.ny, format="csc")
    cubic = (u - model.alpha_1) * (u - model.alpha_2) * (model.alpha_3 - u)
    rhs_u = u + grid.dt * (model.skew * cubic - v + model.potential)
    rhs_v = v + grid.dt * model.gamma * (model.beta * u - v)
    expected_u = spsolve(identity - grid.dt * model.diffusion_u * laplacian, rhs_u.ravel()).reshape(
        u.shape
    )
    expected_v = spsolve(identity - grid.dt * model.diffusion_v * laplacian, rhs_v.ravel()).reshape(
        v.shape
    )

    np.testing.assert_allclose(actual_u, expected_u, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual_v, expected_v, rtol=1e-13, atol=1e-13)


def test_tiny_signal_is_reproducible_and_not_clipped(tiny_config) -> None:
    one = simulate_signal(tiny_config)
    two = simulate_signal(tiny_config)
    np.testing.assert_array_equal(one.activator, two.activator)
    assert np.all(np.isfinite(one.activator))
    assert one.warmup_steps == 1


def test_quick_profile_calibration_regression() -> None:
    from spde_poisson_filtering.config import ExperimentConfig

    signal = simulate_signal(ExperimentConfig.from_yaml("configs/quick.yaml"))
    diagnostics = signal.diagnostics
    assert diagnostics["trajectory_q005"] >= -0.1
    assert diagnostics["trajectory_q995"] <= 1.1
    assert diagnostics["median_spatial_contrast"] >= 0.18
    assert diagnostics["rms_temporal_change"] >= 5.0e-4
