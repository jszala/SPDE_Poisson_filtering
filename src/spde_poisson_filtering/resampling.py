"""Particle-filter diagnostics and resampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_weights(weights: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("at least one particle weight must be positive")
    return values / total


def effective_sample_size(weights: ArrayLike) -> float:
    r"""Return :math:`1 / \sum_i w_i^2` for normalized weights."""

    values = _validated_weights(weights)
    return float(1.0 / np.sum(values * values))


def residual_resample(weights: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    """Residual resampling with the residual mass computed on the right scale."""

    values = _validated_weights(weights)
    count = values.size
    deterministic_copies = np.floor(count * values).astype(np.int64)
    deterministic = np.repeat(np.arange(count, dtype=np.int64), deterministic_copies)
    remaining = count - deterministic.size
    if remaining == 0:
        return deterministic

    # The legacy implementation subtracted ``floor(N*w)`` from ``w``. The
    # correct residual is N*w-floor(N*w), equivalently w-floor(N*w)/N.
    residual = count * values - deterministic_copies
    residual_total = float(np.sum(residual))
    if residual_total <= 0:
        stochastic = rng.integers(0, count, size=remaining, dtype=np.int64)
    else:
        residual /= residual_total
        stochastic = rng.choice(count, size=remaining, replace=True, p=residual)
    return np.concatenate((deterministic, stochastic)).astype(np.int64, copy=False)
