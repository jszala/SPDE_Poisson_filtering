"""Typed experiment configuration and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    dx: float
    dt: float
    steps: int


@dataclass(frozen=True)
class FHNConfig:
    diffusion_u: float
    diffusion_v: float
    alpha_1: float
    alpha_2: float
    alpha_3: float
    beta: float
    gamma: float
    skew: float
    potential: float
    sigma_u: float
    sigma_v: float


@dataclass(frozen=True)
class WarmupConfig:
    min_steps: int
    max_steps: int
    check_every: int
    initial_u_mean: float
    initial_u_std: float
    initial_v_mean: float
    initial_v_std: float
    active_threshold: float
    min_active_fraction: float
    max_active_fraction: float
    min_spatial_contrast: float


@dataclass(frozen=True)
class ObservationConfig:
    coefficient: float
    decay: float
    lambda_min: float
    lambda_max: float
    resolutions: tuple[int, ...]


@dataclass(frozen=True)
class FilterConfig:
    particles: int
    prior_std_u: float
    prior_std_v: float
    ess_fraction: float


@dataclass(frozen=True)
class ExecutionConfig:
    seed: int
    workers: int


@dataclass(frozen=True)
class ExperimentConfig:
    grid: GridConfig
    model: FHNConfig
    warmup: WarmupConfig
    observation: ObservationConfig
    filter: FilterConfig
    execution: ExecutionConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        try:
            observation = dict(payload["observation"])
            observation["resolutions"] = tuple(int(x) for x in observation["resolutions"])
            config = cls(
                grid=GridConfig(**payload["grid"]),
                model=FHNConfig(**payload["model"]),
                warmup=WarmupConfig(**payload["warmup"]),
                observation=ObservationConfig(**observation),
                filter=FilterConfig(**payload["filter"]),
                execution=ExecutionConfig(**payload["execution"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid experiment configuration: {exc}") from exc
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        grid = self.grid
        model = self.model
        warmup = self.warmup
        obs = self.observation
        filt = self.filter

        if min(grid.nx, grid.ny, grid.steps) <= 0 or grid.dx <= 0 or grid.dt <= 0:
            raise ValueError("grid dimensions, steps, dx and dt must be positive")
        if grid.nx != grid.ny:
            raise ValueError("v2 resolution studies require a square grid")
        if not (model.alpha_1 < model.alpha_2 < model.alpha_3):
            raise ValueError("require alpha_1 < alpha_2 < alpha_3")
        nonnegative = (
            model.diffusion_u,
            model.diffusion_v,
            model.gamma,
            model.sigma_u,
            model.sigma_v,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("diffusion, gamma and noise amplitudes must be non-negative")
        if warmup.min_steps < 0 or warmup.max_steps < warmup.min_steps:
            raise ValueError("warm-up step limits are inconsistent")
        if warmup.check_every <= 0:
            raise ValueError("warmup.check_every must be positive")
        if not 0 <= warmup.min_active_fraction < warmup.max_active_fraction <= 1:
            raise ValueError("warm-up active fractions must lie in [0, 1]")
        if obs.coefficient <= 0 or obs.lambda_min <= 0 or obs.lambda_max <= obs.lambda_min:
            raise ValueError("invalid observation intensity bounds or coefficient")
        if obs.decay < 0:
            raise ValueError("observation decay must be non-negative")
        if not obs.resolutions or any(
            resolution <= 0 or grid.nx % resolution != 0 or grid.ny % resolution != 0
            for resolution in obs.resolutions
        ):
            raise ValueError("every resolution must divide both grid dimensions")
        if filt.particles <= 1 or filt.prior_std_u < 0 or filt.prior_std_v < 0:
            raise ValueError("invalid particle count or prior standard deviation")
        if not 0 < filt.ess_fraction <= 1:
            raise ValueError("filter.ess_fraction must lie in (0, 1]")
        if self.execution.workers <= 0:
            raise ValueError("execution.workers must be positive")
