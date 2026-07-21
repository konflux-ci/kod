"""Tests for the build-image pipeline step."""

import os
import subprocess

from unittest.mock import patch

import pytest

from kod.pipeline.build_image import run_build_image


@pytest.fixture()
def build_dir(tmp_path):
    """Create a build context with required index artifacts and chdir into it."""
    index_dir = tmp_path / "data" / "index"
    index_dir.mkdir(parents=True)
    (index_dir / "index.faiss").write_bytes(b"fake")
    (index_dir / "metadata.jsonl").write_bytes(b"fake")
    (tmp_path / "Containerfile").write_text("FROM scratch\n")
    prev = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(prev)


@patch("kod.pipeline.build_image.subprocess.run")
def test_happy_path(mock_run, build_dir):
    run_build_image(data_dir="data")
    mock_run.assert_called_once_with(
        [
            "podman",
            "build",
            "-t",
            "kod:latest",
            "-f",
            "Containerfile",
            "--build-arg",
            "DATA_DIR=data",
            ".",
        ],
        check=True,
    )


@patch("kod.pipeline.build_image.subprocess.run")
def test_custom_builder(mock_run, build_dir):
    run_build_image(data_dir="data", builder="docker")
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "docker"


@patch("kod.pipeline.build_image.subprocess.run")
def test_custom_tag(mock_run, build_dir):
    run_build_image(data_dir="data", tag="myimage:v1")
    cmd = mock_run.call_args[0][0]
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "myimage:v1"


def test_missing_index_faiss(build_dir):
    (build_dir / "data" / "index" / "index.faiss").unlink()
    with pytest.raises(FileNotFoundError, match="index.faiss"):
        run_build_image(data_dir="data")


def test_missing_metadata(build_dir):
    (build_dir / "data" / "index" / "metadata.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.jsonl"):
        run_build_image(data_dir="data")


def test_missing_containerfile(build_dir):
    (build_dir / "Containerfile").unlink()
    with pytest.raises(FileNotFoundError, match="Containerfile not found"):
        run_build_image(data_dir="data")


def test_absolute_data_dir_rejected(build_dir):
    with pytest.raises(ValueError, match="relative path"):
        run_build_image(data_dir="/absolute/path")


def test_parent_traversal_rejected(build_dir):
    with pytest.raises(ValueError, match="relative path"):
        run_build_image(data_dir="../outside")


@patch(
    "kod.pipeline.build_image.subprocess.run",
    side_effect=FileNotFoundError("No such file"),
)
def test_builder_not_found(mock_run, build_dir):
    with pytest.raises(FileNotFoundError, match="not found"):
        run_build_image(data_dir="data")


@patch(
    "kod.pipeline.build_image.subprocess.run",
    side_effect=subprocess.CalledProcessError(1, "podman"),
)
def test_build_failure_propagates(mock_run, build_dir):
    with pytest.raises(subprocess.CalledProcessError):
        run_build_image(data_dir="data")
