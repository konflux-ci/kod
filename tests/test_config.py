"""Tests for KOD configuration loading and validation."""

import textwrap

from pathlib import Path

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


def test_source_include_paths():
    source = DocumentSource(
        name="test",
        url="https://github.com/org/repo.git",
        include_paths=["docs", "content"],
    )
    assert source.include_paths == ["docs", "content"]
    assert source.exclude_paths == []


def test_source_exclude_paths():
    source = DocumentSource(
        name="test",
        url="https://github.com/org/repo.git",
        exclude_paths=["vendor", ".github"],
    )
    assert source.exclude_paths == ["vendor", ".github"]
    assert source.include_paths == []


def test_source_max_pages_default():
    source = DocumentSource(name="test", url="https://example.com")
    assert source.max_pages == 50


def test_source_max_pages_custom():
    source = DocumentSource(name="test", url="https://example.com", max_pages=100)
    assert source.max_pages == 100


def test_source_max_pages_zero_rejected():
    with pytest.raises(ValidationError, match="max_pages"):
        DocumentSource(name="test", url="https://example.com", max_pages=0)


def test_source_use_sitemap_default():
    source = DocumentSource(name="test", url="https://example.com")
    assert source.use_sitemap is True


def test_source_use_sitemap_disabled():
    source = DocumentSource(name="test", url="https://example.com", use_sitemap=False)
    assert source.use_sitemap is False


def test_source_name_with_slash_rejected():
    with pytest.raises(ValidationError, match="name"):
        DocumentSource(name="../../etc", url="https://example.com")


def test_source_name_with_backslash_rejected():
    with pytest.raises(ValidationError, match="name"):
        DocumentSource(name="a\\b", url="https://example.com")


def test_source_name_with_dotdot_rejected():
    with pytest.raises(ValidationError, match="name"):
        DocumentSource(name="a..b", url="https://example.com")


def test_source_name_with_null_rejected():
    with pytest.raises(ValidationError, match="name"):
        DocumentSource(name="a\0b", url="https://example.com")


def test_source_both_paths_rejected():
    with pytest.raises(ValidationError, match="include_paths and exclude_paths"):
        DocumentSource(
            name="test",
            url="https://github.com/org/repo.git",
            include_paths=["docs"],
            exclude_paths=["vendor"],
        )


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


def test_data_dir_default():
    config = KodConfig(sources=[DocumentSource(name="test", url="https://example.com")])
    assert config.data_dir == Path("data")


def test_data_dir_custom(tmp_path):
    custom = tmp_path / "custom"
    config = KodConfig(
        sources=[DocumentSource(name="test", url="https://example.com")],
        data_dir=custom,
    )
    assert config.data_dir == custom


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


def test_chunk_size_default():
    config = KodConfig(sources=[DocumentSource(name="test", url="https://example.com")])
    assert config.chunk_size == 1000


def test_chunk_size_custom():
    config = KodConfig(
        sources=[DocumentSource(name="test", url="https://example.com")],
        chunk_size=500,
    )
    assert config.chunk_size == 500


def test_chunk_overlap_default():
    config = KodConfig(sources=[DocumentSource(name="test", url="https://example.com")])
    assert config.chunk_overlap == 200


def test_chunk_overlap_custom():
    config = KodConfig(
        sources=[DocumentSource(name="test", url="https://example.com")],
        chunk_overlap=100,
    )
    assert config.chunk_overlap == 100


def test_chunk_overlap_equals_chunk_size_rejected():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com")],
            chunk_size=500,
            chunk_overlap=500,
        )


def test_chunk_overlap_exceeds_chunk_size_rejected():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        KodConfig(
            sources=[DocumentSource(name="test", url="https://example.com")],
            chunk_size=500,
            chunk_overlap=600,
        )


def test_embedding_model_default():
    config = KodConfig(sources=[DocumentSource(name="test", url="https://example.com")])
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"


def test_embedding_model_custom():
    config = KodConfig(
        sources=[DocumentSource(name="test", url="https://example.com")],
        embedding_model="BAAI/bge-base-en-v1.5",
    )
    assert config.embedding_model == "BAAI/bge-base-en-v1.5"
