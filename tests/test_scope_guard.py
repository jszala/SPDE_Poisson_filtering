from __future__ import annotations

from pathlib import Path


def test_runtime_package_has_no_remote_or_cluster_dependencies() -> None:
    banned = (
        "mpi4py",
        "webdav",
        "boto3",
        "paramiko",
        "requests.put",
        "aws s3",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in Path("src/spde_poisson_filtering").glob("*.py")
    )
    for token in banned:
        assert token not in source

    metadata = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    for token in banned:
        assert token not in metadata
