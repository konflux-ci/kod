"""Tests for KOD configuration loading and validation."""

import textwrap

import pytest

from pydantic import ValidationError

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.config import load_config


def test_source_with_all_fields():
    source = DocumentSource(
        name="test",
        url="https://github.com/org/repo.git",
        metadata={"product": "Test"},
    )
    assert source.name == "test"
    assert source.url == "https://github.com/org/repo.git"
    assert source.metadata == {"product": "Test"}


def test_source_minimal():
    source = DocumentSource(
        name="test",
        url="https://example.com",
    )
    assert source.metadata == {}


def test_source_missing_name_rejected():
    with pytest.raises(ValidationError, match="name"):
        DocumentSource(url="https://example.com")


def test_source_missing_url_rejected():
    with pytest.raises(ValidationError, match="url"):
        DocumentSource(name="test")


def test_empty_sources_rejected():
    with pytest.raises(ValidationError, match="sources"):
        KodConfig(sources=[])


def test_valid_config():
    config = KodConfig(
        sources=[
            DocumentSource(name="test", url="https://github.com/org/repo.git"),
        ]
    )
    assert len(config.sources) == 1


def test_load_valid_config(sample_config_yaml):
    config = load_config(sample_config_yaml)
    assert len(config.sources) == 1
    assert config.sources[0].name == "test-docs"
    assert config.sources[0].url == "https://github.com/example/docs.git"


def test_load_minimal_config(minimal_config_yaml):
    config = load_config(minimal_config_yaml)
    assert len(config.sources) == 1
    assert config.sources[0].metadata == {}


def test_load_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_invalid_yaml(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("sources: not_a_list")
    with pytest.raises(ValidationError):
        load_config(bad_file)


def test_load_empty_sources(tmp_path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("sources: []")
    with pytest.raises(ValidationError, match="sources"):
        load_config(empty_file)


def test_load_multiple_sources(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent("""\
        sources:
          - name: repo-a
            url: https://github.com/org/a.git
          - name: page-b
            url: https://example.com/docs
    """)
    )
    config = load_config(str(config_file))
    assert len(config.sources) == 2
    assert config.sources[0].name == "repo-a"
    assert config.sources[1].name == "page-b"
