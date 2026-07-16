"""KOD CLI - Konflux Offline Documentation pipeline."""

import logging

import click

from kod.config import KodConfig
from kod.config import load_config


logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _get_config(ctx: click.Context) -> KodConfig:
    """Load and cache the config from the path stored in ctx.obj."""
    if "config" not in ctx.obj:
        try:
            ctx.obj["config"] = load_config(ctx.obj["config_path"])
        except (FileNotFoundError, OSError, ValueError) as e:
            raise click.BadParameter(str(e), param_hint="'--config'") from e
    return ctx.obj["config"]


@click.group()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(dir_okay=False),
    default="config.yaml",
    show_default=True,
    help="Path to the KOD configuration file.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.version_option(package_name="kod")
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """KOD - Konflux Offline Documentation pipeline."""
    _configure_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.command()
@click.pass_context
def extract(ctx: click.Context) -> None:
    """Extract documents from configured sources."""
    from kod.pipeline.extract import run_extract

    run_extract(_get_config(ctx))


@cli.command()
@click.pass_context
def transform(ctx: click.Context) -> None:
    """Chunk extracted documents for embedding."""
    from kod.pipeline.transform import run_transform

    run_transform(_get_config(ctx))


@cli.command()
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Generate embeddings for document chunks."""
    from kod.pipeline.embed import run_embed

    run_embed(_get_config(ctx))


@cli.command()
@click.pass_context
def index(ctx: click.Context) -> None:
    """Build FAISS index from embeddings."""
    from kod.pipeline.index import run_index

    run_index(_get_config(ctx))


@cli.command("build-image")
@click.pass_context
def build_image(ctx: click.Context) -> None:
    """Build the container image with the pre-built index."""
    from kod.pipeline.build_image import run_build_image

    run_build_image(_get_config(ctx))


@cli.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    envvar="KOD_DATA_DIR",
    default="data",
    show_default=True,
    help="Path to the data directory containing the FAISS index.",
)
@click.option(
    "--model",
    envvar="KOD_EMBEDDING_MODEL",
    default="BAAI/bge-small-en-v1.5",
    show_default=True,
    help="FastEmbed model name for query embedding.",
)
@click.option(
    "--rrf-k",
    type=int,
    envvar="KOD_RRF_K",
    default=60,
    show_default=True,
    help="RRF constant k for reciprocal rank fusion.",
)
@click.option(
    "--max-queries",
    type=int,
    envvar="KOD_MAX_QUERIES",
    default=5,
    show_default=True,
    help="Maximum number of query reformulations to accept.",
)
@click.option(
    "--max-top-k",
    type=int,
    envvar="KOD_MAX_TOP_K",
    default=20,
    show_default=True,
    help="Maximum top_k value allowed per search request.",
)
def serve(data_dir: str, model: str, rrf_k: int, max_queries: int, max_top_k: int) -> None:
    """Start the MCP server."""
    from kod.server import run_server

    run_server(
        data_dir=data_dir,
        embedding_model=model,
        rrf_k=rrf_k,
        max_queries=max_queries,
        max_top_k=max_top_k,
    )


@cli.command()
@click.pass_context
def pipeline(ctx: click.Context) -> None:
    """Run the full ETL pipeline: extract -> transform -> embed -> index."""
    from kod.pipeline import run_pipeline

    run_pipeline(_get_config(ctx))


def main() -> None:
    """Entry point for the 'kod' console script."""
    cli()  # pragma: no cover
