"""Tests for KOD CLI subcommands."""

from unittest.mock import patch

import pytest

from click.testing import CliRunner

from kod.cli import cli


@pytest.fixture()
def runner():
    return CliRunner()


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Konflux Offline Documentation" in result.output


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "cmd",
    ["extract", "transform", "embed", "index", "build-image", "serve", "pipeline"],
)
def test_subcommand_help(runner, cmd):
    result = runner.invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "cmd",
    ["transform", "embed", "index"],
)
def test_subcommand_runs_with_config(runner, sample_config_yaml, cmd):
    result = runner.invoke(cli, ["--config", sample_config_yaml, cmd])
    assert result.exit_code == 0, f"Failed for '{cmd}': {result.output}"


@patch("kod.pipeline.extract.run_extract")
def test_extract_runs_with_config(mock_extract, runner, sample_config_yaml):
    result = runner.invoke(cli, ["--config", sample_config_yaml, "extract"])
    assert result.exit_code == 0
    mock_extract.assert_called_once()


@patch("kod.pipeline.extract.run_extract")
def test_pipeline_runs_with_config(mock_extract, runner, sample_config_yaml):
    result = runner.invoke(cli, ["--config", sample_config_yaml, "pipeline"])
    assert result.exit_code == 0
    mock_extract.assert_called_once()


@patch("kod.server.run_server")
def test_serve_runs(mock_run, runner, tmp_path):
    result = runner.invoke(cli, ["serve", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        data_dir=str(tmp_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        rrf_k=60,
        max_queries=5,
        max_top_k=20,
    )


@patch("kod.server.run_server")
def test_serve_model_flag(mock_run, runner, tmp_path):
    result = runner.invoke(cli, ["serve", "--data-dir", str(tmp_path), "--model", "custom/model"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        data_dir=str(tmp_path),
        embedding_model="custom/model",
        rrf_k=60,
        max_queries=5,
        max_top_k=20,
    )


@patch("kod.server.run_server")
def test_serve_tuning_flags(mock_run, runner, tmp_path):
    result = runner.invoke(
        cli,
        [
            "serve",
            "--data-dir",
            str(tmp_path),
            "--rrf-k",
            "30",
            "--max-queries",
            "3",
            "--max-top-k",
            "10",
        ],
    )
    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        data_dir=str(tmp_path),
        embedding_model="BAAI/bge-small-en-v1.5",
        rrf_k=30,
        max_queries=3,
        max_top_k=10,
    )


@patch("kod.pipeline.build_image.run_build_image")
def test_build_image_runs(mock_build, runner):
    result = runner.invoke(cli, ["build-image"])
    assert result.exit_code == 0
    mock_build.assert_called_once()


@patch("kod.pipeline.build_image.run_build_image")
def test_build_image_builder_flag(mock_build, runner):
    result = runner.invoke(cli, ["build-image", "--builder", "docker"])
    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["builder"] == "docker"


@patch("kod.pipeline.build_image.run_build_image")
def test_build_image_tag_flag(mock_build, runner):
    result = runner.invoke(cli, ["build-image", "-t", "myimg:v2"])
    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["tag"] == "myimg:v2"


@patch("kod.pipeline.build_image.run_build_image")
def test_build_image_containerfile_flag(mock_build, runner):
    result = runner.invoke(cli, ["build-image", "--containerfile", "Custom.Dockerfile"])
    assert result.exit_code == 0
    _, kwargs = mock_build.call_args
    assert kwargs["containerfile"] == "Custom.Dockerfile"


@patch("kod.pipeline.build_image.run_build_image")
def test_build_image_defaults(mock_build, runner):
    runner.invoke(cli, ["build-image"])
    _, kwargs = mock_build.call_args
    assert kwargs["data_dir"] == "data"
    assert kwargs["builder"] == "podman"
    assert kwargs["tag"] == "kod:latest"
    assert kwargs["containerfile"] == "Containerfile"


def test_missing_config_fails(runner):
    result = runner.invoke(cli, ["--config", "nonexistent.yaml", "extract"])
    assert result.exit_code != 0


def test_get_config_caches_result():
    from unittest.mock import MagicMock

    from kod.cli import _get_config

    config = MagicMock()
    ctx = MagicMock()
    ctx.obj = {"config": config}
    assert _get_config(ctx) is config


def test_get_config_missing_file_raises():
    from unittest.mock import MagicMock

    import click

    from kod.cli import _get_config

    ctx = MagicMock()
    ctx.obj = {"config_path": "nonexistent.yaml"}
    with pytest.raises(click.BadParameter, match="No such file"):
        _get_config(ctx)


def test_invalid_config_shows_clean_error(runner, tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("sources: not_a_list")
    result = runner.invoke(cli, ["--config", str(bad_config), "extract"])
    assert result.exit_code != 0
    assert "Error" in result.output
