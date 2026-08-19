"""Poisson likelihood primitives shared by simulation and filtering."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

FloatArray = NDArray[np.float64]
Axis = int | tuple[int, ...] | None


def _validated_inputs(
    counts: ArrayLike, intensity: ArrayLike, dt: float
) -> tuple[FloatArray, FloatArray]:
    count_array = np.asarray(counts)
    intensity_array = np.asarray(intensity, dtype=np.float64)
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if not np.all(np.isfinite(count_array)) or np.any(count_array < 0):
        raise ValueError("counts must be finite and non-negative")
    if not np.all(np.equal(count_array, np.floor(count_array))):
        raise ValueError("counts must be integer-valued")
    if not np.all(np.isfinite(intensity_array)) or np.any(intensity_array <= 0):
        raise ValueError("intensity must be finite and strictly positive")
    try:
        np.broadcast_shapes(count_array.shape, intensity_array.shape)
    except ValueError as exc:
        raise ValueError("counts and intensity are not broadcast-compatible") from exc
    return count_array.astype(np.float64, copy=False), intensity_array


def poisson_log_likelihood(
    counts: ArrayLike,
    intensity: ArrayLike,
    dt: float,
    *,
    include_constant: bool = True,
    axis: Axis = None,
) -> FloatArray | np.float64:
    """Return the Poisson log likelihood for rate ``intensity`` over interval ``dt``.

    ``intensity`` is an event rate, not the Poisson mean. The mean is therefore
    ``dt * intensity``. Keeping this distinction explicit prevents the missing-
    ``dt`` errors that are common in point-process filtering code.
    """

    count_array, intensity_array = _validated_inputs(counts, intensity, dt)
    mean = dt * intensity_array
    terms = count_array * np.log(mean) - mean
    if include_constant:
        terms = terms - gammaln(count_array + 1.0)
    return np.sum(terms, axis=axis, dtype=np.float64)


def poisson_log_weight_increment(
    counts: ArrayLike,
    intensity: ArrayLike,
    dt: float,
    *,
    reference_intensity: ArrayLike | float = 1.0,
    axis: Axis = None,
) -> FloatArray | np.float64:
    r"""Return the reference-measure log-weight increment.

    For particle intensity :math:`\lambda` and positive reference rate
    :math:`\lambda_0`, this evaluates

    .. math::
       \sum_i \Delta Y_i\log(\lambda_i/\lambda_{0,i})
       - \Delta t(\lambda_i-\lambda_{0,i}).

    It differs from the full Poisson log likelihood only by a term independent
    of the particle and is consequently safe for normalized particle weights.
    """

    count_array, intensity_array = _validated_inputs(counts, intensity, dt)
    reference = np.asarray(reference_intensity, dtype=np.float64)
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0):
        raise ValueError("reference_intensity must be finite and strictly positive")
    try:
        np.broadcast_shapes(intensity_array.shape, reference.shape)
    except ValueError as exc:
        raise ValueError("reference_intensity is not broadcast-compatible") from exc
    terms = count_array * np.log(intensity_array / reference) - dt * (intensity_array - reference)
    return np.sum(terms, axis=axis, dtype=np.float64)


def normalize_log_weights(log_weights: ArrayLike) -> tuple[FloatArray, float]:
    """Normalize one-dimensional log weights with a stable log-sum-exp."""

    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("particle log weights contain NaN or infinity")
    maximum = float(np.max(values))
    log_normalizer = maximum + float(np.log(np.sum(np.exp(values - maximum))))
    weights = np.exp(values - log_normalizer)
    weights /= np.sum(weights)
    return weights, log_normalizer
