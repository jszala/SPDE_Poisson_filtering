"""Particle filtering for a 2D stochastic FHN model with Poisson observations."""

from .config import ExperimentConfig
from .filter import BootstrapParticleFilter, FilterResult
from .likelihood import poisson_log_likelihood, poisson_log_weight_increment
from .observations import ObservationSet, simulate_observations
from .spde import SignalResult, simulate_signal

__all__ = [
    "BootstrapParticleFilter",
    "ExperimentConfig",
    "FilterResult",
    "ObservationSet",
    "SignalResult",
    "poisson_log_likelihood",
    "poisson_log_weight_increment",
    "simulate_observations",
    "simulate_signal",
]

__version__ = "2.0.0"
