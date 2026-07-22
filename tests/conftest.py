"""Shared fixtures for KOD tests."""

import textwrap

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def sample_config_yaml(tmp_path):
    """Write a valid config YAML and return its path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent("""\
        sources:
          - name: test-docs
            url: https://github.com/example/docs.git
            metadata:
              product: TestProduct
    """)
    )
    return str(config_file)


@pytest.fixture()
def minimal_config_yaml(tmp_path):
    """Config with only required fields."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent("""\
        sources:
          - name: minimal
            url: https://example.com/docs
    """)
    )
    return str(config_file)


def make_ctx(app):
    """Build a minimal FastMCP-compatible context mock."""
    ctx = MagicMock()
    ctx.request_context.lifespan_context = {"app": app}
    return ctx
