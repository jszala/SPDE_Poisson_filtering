from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from spde_poisson_filtering.config import ExperimentConfig


def test_profiles_are_valid() -> None:
    quick = ExperimentConfig.from_yaml("configs/quick.yaml")
    thesis = ExperimentConfig.from_yaml("configs/thesis.yaml")
    assert quick.grid.nx == 32
    assert quick.filter.particles == 512
    assert thesis.grid.nx == 64
    assert thesis.filter.particles == 20_000


def test_resolution_must_divide_grid(tiny_config: ExperimentConfig) -> None:
    invalid = replace(
        tiny_config,
        observation=replace(tiny_config.observation, resolutions=(3,)),
    )
    with pytest.raises(ValueError, match="resolution"):
        invalid.validate()


def test_profiles_are_immutable(tiny_config: ExperimentConfig) -> None:
    with pytest.raises(FrozenInstanceError):
        tiny_config.grid.steps = 10  # type: ignore[misc]
