"""End-to-end synthetic experiment orchestration and reproducible outputs."""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import __version__
from .config import ExperimentConfig
from .filter import BootstrapParticleFilter, FilterResult
from .observations import ObservationSet, simulate_observations
from .spde import SignalResult, simulate_signal


def estimate_peak_memory_bytes(config: ExperimentConfig) -> int:
    """Return a conservative in-memory estimate for a filter run."""

    grid = config.grid
    particles = config.filter.particles
    # Current/propagated u and v, solver RHS/workspace, and a bounded intensity chunk.
    particle_storage = particles * grid.nx * grid.ny * 8 * 6
    trajectory_storage = (grid.steps + 1) * grid.nx * grid.ny * 8 * 6
    return int(particle_storage + trajectory_storage)


def resource_summary(config: ExperimentConfig, workers: int | None = None) -> dict[str, Any]:
    worker_count = workers if workers is not None else config.execution.workers
    peak = estimate_peak_memory_bytes(config)
    return {
        "grid": f"{config.grid.nx}x{config.grid.ny}",
        "steps": config.grid.steps,
        "particles": config.filter.particles,
        "resolutions": list(config.observation.resolutions),
        "workers": worker_count,
        "estimated_peak_gib": round(peak / 1024**3, 2),
    }


def _save_result(path: Path, result: FilterResult) -> None:
    np.savez_compressed(
        path,
        resolution=result.resolution,
        mean_u=result.mean_u,
        mean_v=result.mean_v,
        variance_u=result.variance_u,
        variance_v=result.variance_v,
        ess=result.ess,
        log_normalizers=result.log_normalizers,
        resampled=result.resampled,
        rmse=result.rmse,
        open_loop_rmse=result.open_loop_rmse,
        step_seconds=result.step_seconds,
    )


def _save_metrics(path: Path, results: dict[int, FilterResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=(
                "resolution",
                "mean_rmse",
                "late_rmse",
                "late_open_loop_rmse",
                "resampling_count",
                "mean_ess",
            ),
        )
        writer.writeheader()
        for resolution, result in results.items():
            midpoint = max(1, result.rmse.size // 2)
            writer.writerow(
                {
                    "resolution": resolution,
                    "mean_rmse": float(np.mean(result.rmse[1:])),
                    "late_rmse": float(np.mean(result.rmse[midpoint:])),
                    "late_open_loop_rmse": float(np.mean(result.open_loop_rmse[midpoint:])),
                    "resampling_count": int(np.count_nonzero(result.resampled)),
                    "mean_ess": float(np.mean(result.ess[1:])),
                }
            )


def _save_figures(
    output: Path,
    config: ExperimentConfig,
    signal: SignalResult,
    observations: ObservationSet,
    results: dict[int, FilterResult],
) -> None:
    fine = max(config.observation.resolutions)
    step = min(192, config.grid.steps)
    truth = signal.activator[step]
    counts = observations.counts[fine][step - 1]
    estimate = results[fine].mean_u[step]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    state_limits = (
        float(min(truth.min(), estimate.min())),
        float(max(truth.max(), estimate.max())),
    )
    images = [
        axes[0].imshow(
            truth,
            origin="lower",
            cmap="viridis",
            vmin=state_limits[0],
            vmax=state_limits[1],
        ),
        axes[1].imshow(counts, origin="lower", cmap="magma"),
        axes[2].imshow(
            estimate,
            origin="lower",
            cmap="viridis",
            vmin=state_limits[0],
            vmax=state_limits[1],
        ),
    ]
    axes[0].set_title("True activator")
    axes[1].set_title(f"Poisson counts ({fine}×{fine})")
    axes[2].set_title("Posterior mean")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(images[0], ax=(axes[0], axes[2]), shrink=0.8, label="state")
    figure.colorbar(images[1], ax=axes[1], shrink=0.8, label="count increment")
    figure.savefig(output / "state_observation_filter.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    times = np.arange(config.grid.steps + 1) * config.grid.dt
    for resolution, result in results.items():
        axis.plot(times, result.rmse, label=f"{resolution}×{resolution}", linewidth=1.5)
    axis.plot(
        times,
        results[fine].open_loop_rmse,
        color="black",
        linestyle="--",
        label="open-loop baseline",
    )
    axis.set_xlabel("time")
    axis.set_ylabel("activator RMSE")
    axis.legend(ncol=2)
    figure.savefig(output / "rmse_by_resolution.png", dpi=180)
    plt.close(figure)


def run_experiment(
    config: ExperimentConfig,
    output: str | Path,
    *,
    workers: int | None = None,
    progress: bool = True,
) -> dict[int, FilterResult]:
    """Run truth, observations, and each resolution filter sequentially."""

    if workers is not None:
        config = replace(config, execution=replace(config.execution, workers=workers))
        config.validate()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    signal = simulate_signal(config)
    observations = simulate_observations(signal, config)
    results: dict[int, FilterResult] = {}
    for resolution in config.observation.resolutions:
        particle_filter = BootstrapParticleFilter(
            config, resolution, workers=config.execution.workers
        )
        results[resolution] = particle_filter.run(
            observations.counts[resolution], signal, progress=progress
        )

    np.savez_compressed(
        output_path / "truth.npz",
        activator=signal.activator,
        inhibitor=signal.inhibitor,
        warmup_steps=signal.warmup_steps,
    )
    np.savez_compressed(
        output_path / "observations.npz",
        **{f"counts_{resolution}": values for resolution, values in observations.counts.items()},
    )
    for resolution, result in results.items():
        _save_result(output_path / f"filter_{resolution}.npz", result)
    _save_metrics(output_path / "metrics.csv", results)
    _save_figures(output_path, config, signal, observations, results)

    expected_outputs = [
        "manifest.json",
        "metrics.csv",
        "observations.npz",
        "rmse_by_resolution.png",
        "state_observation_filter.png",
        "truth.npz",
        *(f"filter_{resolution}.npz" for resolution in config.observation.resolutions),
    ]
    manifest = {
        "software_version": __version__,
        "python": platform.python_version(),
        "config": config.to_dict(),
        "resource_estimate": resource_summary(config),
        "signal_diagnostics": signal.diagnostics,
        "outputs": sorted(expected_outputs),
    }
    with (output_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return results


__all__ = [
    "estimate_peak_memory_bytes",
    "resource_summary",
    "run_experiment",
]
