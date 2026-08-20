from __future__ import annotations

import csv
import json

import numpy as np

from spde_poisson_filtering.experiment import resource_summary, run_experiment


def test_resource_estimate_is_positive(tiny_config) -> None:
    summary = resource_summary(tiny_config, workers=2)
    assert summary["workers"] == 2
    assert summary["estimated_peak_gib"] >= 0


def test_end_to_end_outputs(tmp_path, tiny_config) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    results = run_experiment(tiny_config, first, workers=1, progress=False)
    run_experiment(tiny_config, second, workers=1, progress=False)
    assert set(results) == {4, 2}
    expected = {
        "truth.npz",
        "observations.npz",
        "filter_4.npz",
        "filter_2.npz",
        "metrics.csv",
        "state_observation_filter.png",
        "rmse_by_resolution.png",
        "manifest.json",
    }
    assert expected <= {path.name for path in first.iterdir()}
    with (first / "manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["software_version"] == "2.0.0"
    with (first / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(np.isfinite(float(row["mean_rmse"])) for row in rows)

    reproducible = (
        "manifest.json",
        "metrics.csv",
        "state_observation_filter.png",
        "rmse_by_resolution.png",
    )
    for name in reproducible:
        assert (first / name).read_bytes() == (second / name).read_bytes()
