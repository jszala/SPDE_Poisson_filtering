"""Poisson/Cox-process observation construction and spatial aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import ExperimentConfig, ObservationConfig

if TYPE_CHECKING:
    from .spde import SignalResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ObservationSet:
    """Fine observations and Poisson-consistent block-summed resolutions."""

    counts: dict[int, IntArray]
    rates: dict[int, FloatArray]
    fine_resolution: int


def intensity_density(activator: ArrayLike, time: float, config: ObservationConfig) -> FloatArray:
    """Evaluate the bounded, strictly positive pointwise intensity density."""

    state = np.asarray(activator, dtype=np.float64)
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("activator contains NaN or infinity")
    raw = np.exp(-config.decay * time) * (config.coefficient * np.maximum(state, 0.0)) ** 2
    return np.clip(raw, config.lambda_min, config.lambda_max)


def cell_rates(
    activator: ArrayLike, time: float, config: ObservationConfig, dx: float
) -> FloatArray:
    """Integrate the pointwise density over equal square finite-difference cells."""

    if dx <= 0:
        raise ValueError("dx must be positive")
    return intensity_density(activator, time, config) * dx**2


def aggregate_blocks(values: ArrayLike, resolution: int) -> NDArray:
    """Sum square fine-grid blocks while preserving any leading dimensions."""

    array = np.asarray(values)
    if array.ndim < 2 or array.shape[-2] != array.shape[-1]:
        raise ValueError("values must end in square spatial dimensions")
    fine = array.shape[-1]
    if resolution <= 0 or fine % resolution:
        raise ValueError("resolution must be a positive divisor of the fine grid")
    factor = fine // resolution
    shape = (*array.shape[:-2], resolution, factor, resolution, factor)
    return array.reshape(shape).sum(axis=(-3, -1))


def simulate_observations(signal: SignalResult, config: ExperimentConfig) -> ObservationSet:
    """Simulate fine Poisson increments once and aggregate the same events."""

    # Local import avoids a runtime cycle while preserving the public type hint.
    from .spde import SignalResult

    if not isinstance(signal, SignalResult):
        raise TypeError("signal must be a SignalResult")
    grid = config.grid
    if signal.activator.shape != (grid.steps + 1, grid.nx, grid.ny):
        raise ValueError("signal trajectory shape does not match the configuration")

    fine_rates = np.empty((grid.steps, grid.nx, grid.ny), dtype=np.float64)
    for step in range(1, grid.steps + 1):
        fine_rates[step - 1] = cell_rates(
            signal.activator[step], step * grid.dt, config.observation, grid.dx
        )
    rng = np.random.default_rng(np.random.SeedSequence([config.execution.seed, 31]))
    fine_counts = rng.poisson(grid.dt * fine_rates).astype(np.int64)

    counts: dict[int, IntArray] = {}
    rates: dict[int, FloatArray] = {}
    for resolution in config.observation.resolutions:
        counts[resolution] = aggregate_blocks(fine_counts, resolution).astype(np.int64, copy=False)
        rates[resolution] = aggregate_blocks(fine_rates, resolution).astype(np.float64, copy=False)
        if not np.array_equal(
            counts[resolution].sum(axis=(-2, -1)), fine_counts.sum(axis=(-2, -1))
        ):
            raise AssertionError("spatial aggregation did not conserve event counts")
    return ObservationSet(counts=counts, rates=rates, fine_resolution=grid.nx)


__all__ = [
    "ObservationSet",
    "aggregate_blocks",
    "cell_rates",
    "intensity_density",
    "simulate_observations",
]
