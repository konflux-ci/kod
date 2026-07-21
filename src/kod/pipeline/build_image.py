"""Build-image step - build container image with pre-built index."""

import logging
import subprocess

from pathlib import Path


logger = logging.getLogger(__name__)

_DEFAULT_BUILDER = "podman"
_DEFAULT_TAG = "kod:latest"
_DEFAULT_CONTAINERFILE = "Containerfile"
_DEFAULT_DATA_DIR = "data"


def run_build_image(
    data_dir: str = _DEFAULT_DATA_DIR,
    builder: str = _DEFAULT_BUILDER,
    tag: str = _DEFAULT_TAG,
    containerfile: str = _DEFAULT_CONTAINERFILE,
) -> None:
    """Build the container image from pre-built ETL artifacts.

    Validates that the FAISS index and metadata exist in the data
    directory, then runs the container build using the specified
    builder binary (podman, docker, etc.).
    """
    data_path = Path(data_dir)
    if data_path.is_absolute() or ".." in data_path.parts:
        raise ValueError(f"data_dir must be a relative path within the build context: {data_dir}")

    index_dir = data_path / "index"
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "metadata.jsonl"

    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}. Run the ETL pipeline first (kod pipeline)."
            )

    containerfile_path = Path(containerfile)
    if not containerfile_path.exists():
        raise FileNotFoundError(f"Containerfile not found: {containerfile_path}")

    cmd = [
        builder,
        "build",
        "-t",
        tag,
        "-f",
        str(containerfile_path),
        "--build-arg",
        f"DATA_DIR={data_dir}",
        ".",
    ]
    logger.info("[build-image] Running: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Builder '{builder}' not found. Install podman or docker, or specify --builder."
        ) from e

    logger.info("[build-image] Image '%s' built successfully", tag)
