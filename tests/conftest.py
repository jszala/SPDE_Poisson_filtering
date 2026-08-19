from __future__ import annotations

from dataclasses import replace

import pytest

from spde_poisson_filtering.config import ExperimentConfig


@pytest.fixture
def tiny_config() -> ExperimentConfig:
    base = ExperimentConfig.from_yaml("configs/quick.yaml")
    return replace(
        base,
        grid=replace(base.grid, nx=4, ny=4, steps=4, dt=0.001),
        warmup=replace(
            base.warmup,
            min_steps=1,
            max_steps=5,
            check_every=1,
            min_active_fraction=0.0,
            max_active_fraction=1.0,
            min_spatial_contrast=0.0,
        ),
        observation=replace(base.observation, resolutions=(4, 2)),
        filter=replace(base.filter, particles=12, prior_std_u=0.02, prior_std_v=0.01),
        execution=replace(base.execution, seed=12345, workers=1),
    )
