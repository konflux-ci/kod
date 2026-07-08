"""Extract step - fetch documents from configured sources."""

import logging
import shutil
import subprocess
import xml.etree.ElementTree as ET

from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from pathlib import PurePosixPath
from urllib.error import URLError
from urllib.parse import urldefrag
from urllib.parse import urljoin
from urllib.parse import urlparse
from urllib.request import urlopen

from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.models import Document


logger = logging.getLogger(__name__)


def run_extract(config: KodConfig) -> None:
    """Extract documents from all configured sources."""
    extracted_dir = config.data_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for source in config.sources:
        logger.info("[extract] Extracting from '%s' (%s)", source.name, source.url)
        try:
            documents = _extract_source(source, config.data_dir, config.doc_extensions)
            output_path = extracted_dir / f"{source.name}.jsonl"
            _write_documents(documents, output_path)
            logger.info(
                "[extract] Wrote %d document(s) to %s",
                len(documents),
                output_path,
            )
        except Exception:
            logger.exception("[extract] Failed to extract '%s', skipping", source.name)
            failures.append(source.name)

    if failures:
        names = ", ".join(failures)
        logger.error("[extract] Extraction finished with %d failure(s): %s", len(failures), names)
    else:
        logger.info("[extract] Extraction complete")


def _is_git_url(url: str) -> bool:
    """Check whether a URL points to a git repository."""
    return url.rstrip("/").endswith(".git")


def _extract_source(
    source: DocumentSource, data_dir: Path, doc_extensions: set[str]
) -> list[Document]:
    """Dispatch extraction to git or web handler based on URL suffix."""
    if _is_git_url(source.url):
        return _extract_git_source(source, data_dir, doc_extensions)
    return _extract_web_source(source)


def _extract_git_source(
    source: DocumentSource, data_dir: Path, doc_extensions: set[str]
) -> list[Document]:
    """Clone a git repo and extract documents from matching files."""
    clone_dir = data_dir / "sources" / source.name
    _clone_repo(source.url, clone_dir)

    documents = []
    for file_path in _find_doc_files(
        clone_dir, doc_extensions, source.include_paths, source.exclude_paths
    ):
        # Use "fast" strategy to skip OCR and expensive model-based parsing
        elements = partition(filename=str(file_path), strategy="fast")
        text = _elements_to_text(elements)
        if not text:
            continue
        rel_path = str(file_path.relative_to(clone_dir))
        documents.append(
            Document(
                content=text,
                source_name=source.name,
                source_url=source.url,
                file_path=rel_path,
                metadata=dict(source.metadata),
            )
        )

    return documents


def _normalize_url(url: str) -> str:
    """Strip the fragment from a URL for deduplication."""
    defragged, _ = urldefrag(url)
    return defragged


class _LinkParser(HTMLParser):
    """Minimal HTML parser that collects href values from anchor tags."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def _extract_links(html: str, base_url: str) -> list[str]:
    """Parse HTML and return deduplicated same-domain HTTP links."""
    parser = _LinkParser()
    parser.feed(html)
    base_netloc = urlparse(base_url).netloc
    seen = set()
    result = []
    for href in parser.links:
        # Resolve relative URLs and strip fragments
        url = _normalize_url(urljoin(base_url, href))
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        # Only follow links on the same domain
        if parsed.netloc != base_netloc:
            continue
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _is_under_path(url: str, base_path: str) -> bool:
    """Check whether a URL's path is equal to or a child of base_path."""
    path = urlparse(url).path
    if base_path in ("", "/"):
        return True
    # Trailing slash stripped to avoid "/docs" not matching "/docs/guide"
    return path == base_path or path.startswith(base_path.rstrip("/") + "/")


def _discover_urls_from_sitemap(base_url: str, max_pages: int) -> list[str] | None:
    """Fetch sitemap.xml and return URLs under the base path, or None if unavailable."""
    parsed = urlparse(base_url)
    base_path = parsed.path
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    try:
        with urlopen(sitemap_url, timeout=30) as resp:  # noqa: S310
            data = resp.read()
    except (URLError, OSError):
        return None
    try:
        root = ET.fromstring(data)  # noqa: S314
    except ET.ParseError:
        return None
    # Try namespaced lookup first, fall back to explicit namespace URI
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = root.findall(".//sm:loc", ns)
    if not locs:
        locs = root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    if not locs:
        return None
    base_netloc = parsed.netloc
    urls = []
    for loc in locs:
        url = _normalize_url(loc.text.strip()) if loc.text else None
        if not url or urlparse(url).netloc != base_netloc:
            continue
        if not _is_under_path(url, base_path):
            continue
        urls.append(url)
        if len(urls) >= max_pages:
            break
    return urls if urls else None


def _discover_urls_by_crawling(seed_url: str, max_pages: int) -> list[tuple[str, str]]:
    """BFS-crawl from seed URL, returning (url, html) pairs scoped to the seed path."""
    seed = _normalize_url(seed_url)
    base_path = urlparse(seed).path
    visited: set[str] = set()
    queue: deque[str] = deque([seed])
    results: list[tuple[str, str]] = []
    while queue and len(results) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            with urlopen(url, timeout=30) as resp:  # noqa: S310
                html = resp.read().decode("utf-8", errors="replace")
        except (URLError, OSError):
            logger.warning("[extract] Failed to fetch %s, skipping", url)
            continue
        results.append((url, html))
        # Only follow links that stay under the seed URL's path
        for link in _extract_links(html, url):
            if link not in visited and _is_under_path(link, base_path):
                queue.append(link)
    return results


def _extract_web_source(source: DocumentSource) -> list[Document]:
    """Extract documents from a web source using sitemap or link crawling."""
    urls = None
    if source.use_sitemap:
        urls = _discover_urls_from_sitemap(source.url, source.max_pages)
    if urls is not None:
        logger.info("[extract] Found %d URL(s) from sitemap", len(urls))
        return _extract_web_pages_by_url(source, urls)

    logger.info("[extract] No sitemap found, crawling links from %s", source.url)
    crawled = _discover_urls_by_crawling(source.url, source.max_pages)
    logger.info("[extract] Crawled %d page(s)", len(crawled))
    return _extract_web_pages_from_crawl(source, crawled)


def _extract_web_pages_by_url(source: DocumentSource, urls: list[str]) -> list[Document]:
    """Partition each URL individually (used with sitemap-discovered URLs)."""
    documents = []
    for url in urls:
        elements = partition_html(url=url)
        text = _elements_to_text(elements)
        if not text:
            continue
        documents.append(
            Document(
                content=text,
                source_name=source.name,
                source_url=source.url,
                file_path=urlparse(url).path,
                metadata=dict(source.metadata),
            )
        )
    return documents


def _extract_web_pages_from_crawl(
    source: DocumentSource, crawled: list[tuple[str, str]]
) -> list[Document]:
    """Partition pre-fetched HTML from crawl results (avoids double-fetching)."""
    documents = []
    for url, html in crawled:
        # Use text= instead of url= since HTML was already fetched during crawl
        elements = partition_html(text=html)
        text = _elements_to_text(elements)
        if not text:
            continue
        documents.append(
            Document(
                content=text,
                source_name=source.name,
                source_url=source.url,
                file_path=urlparse(url).path,
                metadata=dict(source.metadata),
            )
        )
    return documents


def _clone_repo(url: str, target: Path) -> None:
    """Shallow-clone a git repo into target, replacing any existing clone."""
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", url, str(target)],  # noqa: S607
        check=True,
        capture_output=True,
    )


def _find_doc_files(
    directory: Path,
    doc_extensions: set[str],
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> list[Path]:
    """Walk directory for files matching doc_extensions, filtered by include/exclude paths."""
    files = (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in doc_extensions)
    if include_paths:
        files = (
            p
            for p in files
            if any(
                PurePosixPath(p.relative_to(directory)).is_relative_to(ip) for ip in include_paths
            )
        )
    elif exclude_paths:
        files = (
            p
            for p in files
            if not any(
                PurePosixPath(p.relative_to(directory)).is_relative_to(ep) for ep in exclude_paths
            )
        )
    return sorted(files)


def _elements_to_text(elements: list) -> str:
    """Join non-empty Unstructured element texts with double newlines."""
    return "\n\n".join(e.text for e in elements if e.text and e.text.strip())


def _write_documents(documents: list[Document], path: Path) -> None:
    """Serialize documents to a JSONL file, one JSON object per line."""
    with path.open("w") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
