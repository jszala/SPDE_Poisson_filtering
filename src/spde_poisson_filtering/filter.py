"""Bootstrap particle filter for the hidden two-field FHN state."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from tqdm import trange

from .config import ExperimentConfig
from .likelihood import normalize_log_weights, poisson_log_weight_increment
from .observations import aggregate_blocks, cell_rates
from .resampling import effective_sample_size, residual_resample
from .spde import FHNStepper, ParticlePropagator, SignalResult, deterministic_seed

FloatArray = NDArray[np.float64]
_STAGE_OPEN_LOOP = 23


@dataclass(frozen=True)
class FilterResult:
    resolution: int
    mean_u: FloatArray
    mean_v: FloatArray
    variance_u: FloatArray
    variance_v: FloatArray
    ess: FloatArray
    log_normalizers: FloatArray
    resampled: NDArray[np.bool_]
    rmse: FloatArray
    open_loop_rmse: FloatArray
    step_seconds: FloatArray

    @property
    def resampling_steps(self) -> NDArray[np.int64]:
        return np.flatnonzero(self.resampled).astype(np.int64)


class BootstrapParticleFilter:
    """State particle filter with log weights and ESS-triggered residual resampling."""

    def __init__(
        self,
        config: ExperimentConfig,
        resolution: int,
        *,
        workers: int | None = None,
    ):
        config.validate()
        if resolution not in config.observation.resolutions:
            raise ValueError("resolution is not present in the experiment configuration")
        self.config = config
        self.resolution = resolution
        self.workers = workers if workers is not None else config.execution.workers

    def _initialize_particles(
        self, initial_u: FloatArray, initial_v: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        config = self.config
        count = config.filter.particles
        rng = np.random.default_rng(
            np.random.SeedSequence([config.execution.seed, 20, self.resolution])
        )
        shape = (count, config.grid.nx, config.grid.ny)
        u = np.broadcast_to(initial_u, shape).copy()
        v = np.broadcast_to(initial_v, shape).copy()
        if config.filter.prior_std_u:
            u += config.filter.prior_std_u * rng.standard_normal(shape)
        if config.filter.prior_std_v:
            v += config.filter.prior_std_v * rng.standard_normal(shape)
        return u, v

    @staticmethod
    def _moments(values: FloatArray, weights: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean = np.einsum("p,pij->ij", weights, values, optimize=True)
        variance = np.einsum("p,pij->ij", weights, (values - mean) ** 2, optimize=True)
        return mean, variance

    def _weight_increment(
        self,
        particles_u: FloatArray,
        counts: NDArray[np.int64],
        time: float,
    ) -> FloatArray:
        """Compute weights in chunks to avoid a full fine-grid intensity copy."""

        count = particles_u.shape[0]
        increment = np.empty(count, dtype=np.float64)
        chunk_size = min(256, count)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            fine_rates = cell_rates(
                particles_u[start:stop],
                time,
                self.config.observation,
                self.config.grid.dx,
            )
            rates = aggregate_blocks(fine_rates, self.resolution)
            increment[start:stop] = poisson_log_weight_increment(
                counts,
                rates,
                self.config.grid.dt,
                reference_intensity=1.0,
                axis=(-2, -1),
            )
        return increment

    def run(
        self,
        counts: NDArray[np.int64],
        truth: SignalResult,
        *,
        progress: bool = False,
    ) -> FilterResult:
        """Run the filter for one observation resolution."""

        config = self.config
        grid = config.grid
        expected_counts = (grid.steps, self.resolution, self.resolution)
        if counts.shape != expected_counts:
            raise ValueError(f"counts must have shape {expected_counts}")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("Poisson count increments must be non-negative integers")
        if truth.activator.shape != (grid.steps + 1, grid.nx, grid.ny):
            raise ValueError("truth trajectory does not match the configured grid")

        particles_u, particles_v = self._initialize_particles(
            truth.activator[0], truth.inhibitor[0]
        )
        particle_count = config.filter.particles
        weights = np.full(particle_count, 1.0 / particle_count, dtype=np.float64)
        log_weights = np.full(particle_count, -np.log(particle_count), dtype=np.float64)

        trajectory_shape = (grid.steps + 1, grid.nx, grid.ny)
        mean_u = np.empty(trajectory_shape, dtype=np.float64)
        mean_v = np.empty_like(mean_u)
        variance_u = np.empty_like(mean_u)
        variance_v = np.empty_like(mean_u)
        ess = np.empty(grid.steps + 1, dtype=np.float64)
        log_normalizers = np.zeros(grid.steps + 1, dtype=np.float64)
        resampled = np.zeros(grid.steps + 1, dtype=np.bool_)
        rmse = np.empty(grid.steps + 1, dtype=np.float64)
        open_loop_rmse = np.empty(grid.steps + 1, dtype=np.float64)
        step_seconds = np.zeros(grid.steps + 1, dtype=np.float64)

        mean_u[0], variance_u[0] = self._moments(particles_u, weights)
        mean_v[0], variance_v[0] = self._moments(particles_v, weights)
        ess[0] = effective_sample_size(weights)
        rmse[0] = float(np.sqrt(np.mean((mean_u[0] - truth.activator[0]) ** 2)))

        # A seeded, unobserved forecast drawn from the same prior is the relevant
        # baseline. Starting a deterministic forecast at the posterior mean would
        # leak the known synthetic initial state and make the comparison vacuous.
        open_stepper = FHNStepper(grid, config.model)
        open_rng = np.random.default_rng(np.random.SeedSequence([config.execution.seed, 20, 0]))
        open_u = truth.activator[0:1].copy()
        open_v = truth.inhibitor[0:1].copy()
        open_u += config.filter.prior_std_u * open_rng.standard_normal(open_u.shape)
        open_v += config.filter.prior_std_v * open_rng.standard_normal(open_v.shape)
        open_loop_rmse[0] = float(np.sqrt(np.mean((open_u[0] - truth.activator[0]) ** 2)))

        with ParticlePropagator(config, workers=self.workers) as propagator:
            iterator = trange(
                1,
                grid.steps + 1,
                disable=not progress,
                desc=f"PF {self.resolution}x{self.resolution}",
            )
            for step in iterator:
                started = perf_counter()
                particles_u, particles_v = propagator.propagate(particles_u, particles_v, step)
                open_seed = np.array(
                    [
                        deterministic_seed(
                            config.execution.seed,
                            _STAGE_OPEN_LOOP,
                            step,
                            0,
                        )
                    ],
                    dtype=np.uint64,
                )
                open_u, open_v = open_stepper.step(open_u, open_v, seeds=open_seed)

                log_weights += self._weight_increment(particles_u, counts[step - 1], step * grid.dt)
                weights, log_normalizers[step] = normalize_log_weights(log_weights)
                # Keep the normalized representation in log space. Converting the
                # probabilities back through log(exp(.)) would turn harmless
                # floating-point underflow into -inf and poison the next update.
                log_weights -= log_normalizers[step]
                ess[step] = effective_sample_size(weights)

                mean_u[step], variance_u[step] = self._moments(particles_u, weights)
                mean_v[step], variance_v[step] = self._moments(particles_v, weights)
                rmse[step] = float(np.sqrt(np.mean((mean_u[step] - truth.activator[step]) ** 2)))
                open_loop_rmse[step] = float(
                    np.sqrt(np.mean((open_u[0] - truth.activator[step]) ** 2))
                )

                if ess[step] < config.filter.ess_fraction * particle_count:
                    rng = np.random.default_rng(
                        deterministic_seed(config.execution.seed, 22, step, self.resolution)
                    )
                    indices = residual_resample(weights, rng)
                    particles_u = particles_u[indices].copy()
                    particles_v = particles_v[indices].copy()
                    weights.fill(1.0 / particle_count)
                    log_weights.fill(-np.log(particle_count))
                    resampled[step] = True
                step_seconds[step] = perf_counter() - started

        return FilterResult(
            resolution=self.resolution,
            mean_u=mean_u,
            mean_v=mean_v,
            variance_u=variance_u,
            variance_v=variance_v,
            ess=ess,
            log_normalizers=log_normalizers,
            resampled=resampled,
            rmse=rmse,
            open_loop_rmse=open_loop_rmse,
            step_seconds=step_seconds,
        )


__all__ = ["BootstrapParticleFilter", "FilterResult"]
