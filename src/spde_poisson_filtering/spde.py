"""Semi-implicit finite-difference solver for the stochastic 2D FHN system."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

from .config import ExperimentConfig, FHNConfig, GridConfig

FloatArray = NDArray[np.float64]

_STAGE_WARMUP = 11
_STAGE_SIGNAL = 12
_STAGE_PARTICLE = 21


@dataclass(frozen=True)
class SignalResult:
    activator: FloatArray
    inhibitor: FloatArray
    warmup_steps: int
    diagnostics: dict[str, float]


def neumann_laplacian_2d(nx: int, ny: int, dx: float) -> csc_matrix:
    """Return the symmetric finite-volume Neumann Laplacian on a cell grid."""

    if nx <= 1 or ny <= 1 or dx <= 0:
        raise ValueError("nx and ny must exceed one and dx must be positive")

    def one_dimensional(size: int) -> csc_matrix:
        diagonal = -2.0 * np.ones(size)
        diagonal[[0, -1]] = -1.0
        operator = diags(
            (np.ones(size - 1), diagonal, np.ones(size - 1)),
            offsets=(-1, 0, 1),
            format="csc",
        )
        return operator / dx**2

    lx = one_dimensional(nx)
    ly = one_dimensional(ny)
    # NumPy's C-order flattening makes the last (y) coordinate vary fastest.
    return (kron(lx, eye(ny, format="csc")) + kron(eye(nx, format="csc"), ly)).tocsc()


def deterministic_seed(master_seed: int, stage: int, timestep: int, index: int) -> int:
    """Derive a backend-independent seed from semantic coordinates."""

    sequence = np.random.SeedSequence([int(master_seed), int(stage), int(timestep), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


class FHNStepper:
    """Fixed-step semi-implicit Euler-Maruyama propagator."""

    def __init__(self, grid: GridConfig, model: FHNConfig):
        self.grid = grid
        self.model = model
        self.dimension = grid.nx * grid.ny
        laplacian = neumann_laplacian_2d(grid.nx, grid.ny, grid.dx)
        identity = eye(self.dimension, format="csc")
        self._lu_u = splu(identity - grid.dt * model.diffusion_u * laplacian)
        self._lu_v = splu(identity - grid.dt * model.diffusion_v * laplacian)

    def _validate_states(self, activator: FloatArray, inhibitor: FloatArray) -> None:
        if activator.ndim != 3:
            raise ValueError("states must have shape (batch, nx, ny)")
        expected = (activator.shape[0], self.grid.nx, self.grid.ny)
        if activator.shape != expected or inhibitor.shape != expected:
            raise ValueError("states must have shape (batch, nx, ny)")
        if not np.all(np.isfinite(activator)) or not np.all(np.isfinite(inhibitor)):
            raise FloatingPointError("FHN state contains NaN or infinity")

    def step(
        self,
        activator: FloatArray,
        inhibitor: FloatArray,
        seeds: NDArray[np.uint64] | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Advance a batch by one step without modifying its inputs."""

        u = np.asarray(activator, dtype=np.float64)
        v = np.asarray(inhibitor, dtype=np.float64)
        self._validate_states(u, v)
        batch = u.shape[0]
        if seeds is not None and np.asarray(seeds).shape != (batch,):
            raise ValueError("one random seed is required for each particle")

        model = self.model
        cubic = (u - model.alpha_1) * (u - model.alpha_2) * (model.alpha_3 - u)
        reaction_u = model.skew * cubic - v + model.potential
        reaction_v = model.gamma * (model.beta * u - v)

        noise_u = np.zeros_like(u)
        noise_v = np.zeros_like(v)
        if seeds is not None and (model.sigma_u > 0 or model.sigma_v > 0):
            for particle, seed in enumerate(np.asarray(seeds, dtype=np.uint64)):
                rng = np.random.default_rng(int(seed))
                if model.sigma_u > 0:
                    noise_u[particle] = rng.standard_normal((self.grid.nx, self.grid.ny))
                if model.sigma_v > 0:
                    noise_v[particle] = rng.standard_normal((self.grid.nx, self.grid.ny))

        # Space-time white noise in two dimensions has the finite-volume scale
        # sqrt(dt) / dx. No state clipping is applied before or after this step.
        scale = np.sqrt(self.grid.dt) / self.grid.dx
        rhs_u = u + self.grid.dt * reaction_u + model.sigma_u * scale * noise_u
        rhs_v = v + self.grid.dt * reaction_v + model.sigma_v * scale * noise_v
        next_u = self._lu_u.solve(rhs_u.reshape(batch, self.dimension).T).T.reshape(u.shape)
        next_v = self._lu_v.solve(rhs_v.reshape(batch, self.dimension).T).T.reshape(v.shape)

        if not np.all(np.isfinite(next_u)) or not np.all(np.isfinite(next_v)):
            raise FloatingPointError("FHN propagation diverged to NaN or infinity")
        return next_u, next_v


def _excitation_diagnostics(activator: FloatArray, config: ExperimentConfig) -> dict[str, float]:
    warmup = config.warmup
    lower = float(np.quantile(activator, 0.05))
    upper = float(np.quantile(activator, 0.95))
    return {
        "active_fraction": float(np.mean(activator >= warmup.active_threshold)),
        "spatial_contrast": upper - lower,
        "q005": float(np.quantile(activator, 0.005)),
        "q995": float(np.quantile(activator, 0.995)),
    }


def _is_excited(diagnostics: dict[str, float], config: ExperimentConfig) -> bool:
    warmup = config.warmup
    return (
        warmup.min_active_fraction <= diagnostics["active_fraction"] <= warmup.max_active_fraction
        and diagnostics["spatial_contrast"] >= warmup.min_spatial_contrast
        and diagnostics["q005"] >= -0.1
        and diagnostics["q995"] <= 1.1
    )


def _initial_state(config: ExperimentConfig) -> tuple[FloatArray, FloatArray]:
    grid = config.grid
    warmup = config.warmup
    rng = np.random.default_rng(np.random.SeedSequence([config.execution.seed, _STAGE_WARMUP, 0]))
    u = warmup.initial_u_mean + warmup.initial_u_std * gaussian_filter(
        rng.standard_normal((grid.nx, grid.ny)), sigma=1.25, mode="reflect"
    )
    v = warmup.initial_v_mean + warmup.initial_v_std * gaussian_filter(
        rng.standard_normal((grid.nx, grid.ny)), sigma=1.25, mode="reflect"
    )
    return u[np.newaxis, ...], v[np.newaxis, ...]


def simulate_signal(config: ExperimentConfig) -> SignalResult:
    """Warm up an excited state, then simulate the unchanged FHN dynamics."""

    config.validate()
    grid = config.grid
    stepper = FHNStepper(grid, config.model)
    u, v = _initial_state(config)
    diagnostics = _excitation_diagnostics(u[0], config)
    warmup_steps = 0
    for step in range(1, config.warmup.max_steps + 1):
        seeds = np.array(
            [deterministic_seed(config.execution.seed, _STAGE_WARMUP, step, 0)],
            dtype=np.uint64,
        )
        u, v = stepper.step(u, v, seeds)
        warmup_steps = step
        if step >= config.warmup.min_steps and step % config.warmup.check_every == 0:
            diagnostics = _excitation_diagnostics(u[0], config)
            if _is_excited(diagnostics, config):
                break
    else:
        raise RuntimeError(
            "warm-up did not reach the configured excited-state criterion; "
            f"last diagnostics={diagnostics}"
        )

    activator = np.empty((grid.steps + 1, grid.nx, grid.ny), dtype=np.float64)
    inhibitor = np.empty_like(activator)
    activator[0], inhibitor[0] = u[0], v[0]
    for step in range(1, grid.steps + 1):
        seeds = np.array(
            [deterministic_seed(config.execution.seed, _STAGE_SIGNAL, step, 0)],
            dtype=np.uint64,
        )
        u, v = stepper.step(u, v, seeds)
        activator[step], inhibitor[step] = u[0], v[0]

    differences = np.diff(activator, axis=0)
    diagnostics = {
        **diagnostics,
        "trajectory_q005": float(np.quantile(activator, 0.005)),
        "trajectory_q995": float(np.quantile(activator, 0.995)),
        "median_spatial_contrast": float(
            np.median(
                np.quantile(activator, 0.95, axis=(-2, -1))
                - np.quantile(activator, 0.05, axis=(-2, -1))
            )
        ),
        "rms_temporal_change": float(np.sqrt(np.mean(differences * differences))),
    }
    return SignalResult(
        activator=activator,
        inhibitor=inhibitor,
        warmup_steps=warmup_steps,
        diagnostics=diagnostics,
    )


_WORKER_STEPPER: FHNStepper | None = None


def _initialize_worker(grid_payload: dict[str, Any], model_payload: dict[str, Any]) -> None:
    global _WORKER_STEPPER
    _WORKER_STEPPER = FHNStepper(GridConfig(**grid_payload), FHNConfig(**model_payload))


def _worker_step(
    payload: tuple[FloatArray, FloatArray, NDArray[np.uint64]],
) -> tuple[FloatArray, FloatArray]:
    if _WORKER_STEPPER is None:
        raise RuntimeError("particle worker was not initialized")
    return _WORKER_STEPPER.step(*payload)


class ParticlePropagator:
    """Persistent deterministic serial or local-process particle propagation."""

    def __init__(self, config: ExperimentConfig, workers: int | None = None):
        self.config = config
        self.workers = workers if workers is not None else config.execution.workers
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        self._serial = FHNStepper(config.grid, config.model)
        self._pool: ProcessPoolExecutor | None = None
        if self.workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(asdict(config.grid), asdict(config.model)),
            )

    def propagate(
        self, activator: FloatArray, inhibitor: FloatArray, timestep: int
    ) -> tuple[FloatArray, FloatArray]:
        particle_count = activator.shape[0]
        seeds = np.fromiter(
            (
                deterministic_seed(self.config.execution.seed, _STAGE_PARTICLE, timestep, particle)
                for particle in range(particle_count)
            ),
            dtype=np.uint64,
            count=particle_count,
        )
        if self._pool is None:
            return self._serial.step(activator, inhibitor, seeds)

        index_chunks = [
            chunk for chunk in np.array_split(np.arange(particle_count), self.workers) if chunk.size
        ]
        futures = [
            self._pool.submit(
                _worker_step,
                (activator[indices], inhibitor[indices], seeds[indices]),
            )
            for indices in index_chunks
        ]
        results = [future.result() for future in futures]
        return (
            np.concatenate([result[0] for result in results], axis=0),
            np.concatenate([result[1] for result in results], axis=0),
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> ParticlePropagator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "FHNStepper",
    "ParticlePropagator",
    "SignalResult",
    "deterministic_seed",
    "neumann_laplacian_2d",
    "simulate_signal",
]
