"""Tests for KOD document extraction."""

import json

from io import BytesIO
from unittest.mock import patch
from urllib.error import URLError

from kod.config import DocumentSource
from kod.config import KodConfig
from kod.models import Document
from kod.pipeline.extract import _clone_repo
from kod.pipeline.extract import _discover_urls_by_crawling
from kod.pipeline.extract import _discover_urls_from_sitemap
from kod.pipeline.extract import _elements_to_text
from kod.pipeline.extract import _extract_git_source
from kod.pipeline.extract import _extract_links
from kod.pipeline.extract import _extract_source
from kod.pipeline.extract import _extract_web_source
from kod.pipeline.extract import _find_doc_files
from kod.pipeline.extract import _is_git_url
from kod.pipeline.extract import _is_under_path
from kod.pipeline.extract import _normalize_url
from kod.pipeline.extract import _write_documents
from kod.pipeline.extract import run_extract


class FakeElement:
    def __init__(self, text):
        self.text = text


def _git_source(**overrides):
    defaults = {
        "name": "test-repo",
        "url": "https://github.com/org/repo.git",
        "metadata": {"product": "Test"},
    }
    defaults.update(overrides)
    return DocumentSource(**defaults)


def _web_source(**overrides):
    defaults = {
        "name": "test-web",
        "url": "https://example.com/docs",
    }
    defaults.update(overrides)
    return DocumentSource(**defaults)


# --- _is_git_url ---


def test_is_git_url_with_git_suffix():
    assert _is_git_url("https://github.com/org/repo.git") is True


def test_is_git_url_without_git_suffix():
    assert _is_git_url("https://example.com/docs") is False


def test_is_git_url_trailing_slash():
    assert _is_git_url("https://github.com/org/repo.git/") is True


# --- _find_doc_files ---


def test_find_doc_files(tmp_path):
    (tmp_path / "readme.md").touch()
    (tmp_path / "page.html").touch()
    (tmp_path / "other.htm").touch()
    (tmp_path / "code.py").touch()
    (tmp_path / "data.json").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").touch()

    result = _find_doc_files(tmp_path, {".md", ".html", ".htm"})

    names = [p.name for p in result]
    assert "readme.md" in names
    assert "page.html" in names
    assert "other.htm" in names
    assert "nested.md" in names
    assert "code.py" not in names
    assert "data.json" not in names


def test_find_doc_files_sorted(tmp_path):
    (tmp_path / "b.md").touch()
    (tmp_path / "a.md").touch()

    result = _find_doc_files(tmp_path, {".md"})
    assert result[0].name == "a.md"
    assert result[1].name == "b.md"


def test_find_doc_files_include_paths(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").touch()
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.md").touch()
    (tmp_path / "root.md").touch()

    result = _find_doc_files(tmp_path, {".md"}, include_paths=["docs"])
    names = [p.name for p in result]
    assert "guide.md" in names
    assert "notes.md" not in names
    assert "root.md" not in names


def test_find_doc_files_exclude_paths(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").touch()
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "lib.md").touch()
    (tmp_path / "root.md").touch()

    result = _find_doc_files(tmp_path, {".md"}, exclude_paths=["vendor"])
    names = [p.name for p in result]
    assert "guide.md" in names
    assert "root.md" in names
    assert "lib.md" not in names


def test_find_doc_files_no_paths_walks_all(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").touch()
    (tmp_path / "root.md").touch()

    result = _find_doc_files(tmp_path, {".md"})
    assert len(result) == 2


# --- _elements_to_text ---


def test_elements_to_text():
    elements = [FakeElement("Hello"), FakeElement("World")]
    assert _elements_to_text(elements) == "Hello\n\nWorld"


def test_elements_to_text_skips_empty():
    elements = [FakeElement("Hello"), FakeElement(""), FakeElement("World")]
    assert _elements_to_text(elements) == "Hello\n\nWorld"


def test_elements_to_text_skips_whitespace():
    elements = [FakeElement("Hello"), FakeElement("   "), FakeElement("World")]
    assert _elements_to_text(elements) == "Hello\n\nWorld"


def test_elements_to_text_empty_list():
    assert _elements_to_text([]) == ""


def test_elements_to_text_skips_none():
    elements = [FakeElement(None), FakeElement("Hello")]
    assert _elements_to_text(elements) == "Hello"


# --- _write_documents ---


def test_write_documents(tmp_path):
    docs = [
        Document(
            content="Hello",
            source_name="src",
            source_url="https://example.com",
            file_path="a.md",
            metadata={"k": "v"},
        ),
        Document(
            content="World",
            source_name="src",
            source_url="https://example.com",
        ),
    ]
    path = tmp_path / "out.jsonl"
    _write_documents(docs, path)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["content"] == "Hello"
    assert parsed["file_path"] == "a.md"
    assert parsed["metadata"] == {"k": "v"}


def test_write_documents_roundtrip(tmp_path):
    doc = Document(
        content="Test",
        source_name="s",
        source_url="https://example.com",
    )
    path = tmp_path / "out.jsonl"
    _write_documents([doc], path)

    line = path.read_text().strip()
    restored = Document.model_validate_json(line)
    assert restored == doc


def test_write_documents_empty(tmp_path):
    path = tmp_path / "out.jsonl"
    _write_documents([], path)
    assert path.read_text() == ""


# --- _clone_repo ---


@patch("kod.pipeline.extract.subprocess.run")
def test_clone_repo(mock_run, tmp_path):
    target = tmp_path / "repo"
    _clone_repo("https://github.com/org/repo.git", target)

    mock_run.assert_called_once_with(
        ["git", "clone", "--depth", "1", "https://github.com/org/repo.git", str(target)],
        check=True,
        capture_output=True,
    )


@patch("kod.pipeline.extract.subprocess.run")
def test_clone_repo_removes_existing(mock_run, tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    (target / "old-file.txt").write_text("old")

    _clone_repo("https://github.com/org/repo.git", target)
    assert not (target / "old-file.txt").exists()


@patch("kod.pipeline.extract.subprocess.run")
def test_clone_repo_creates_parent(mock_run, tmp_path):
    target = tmp_path / "deep" / "nested" / "repo"
    _clone_repo("https://github.com/org/repo.git", target)
    assert target.parent.exists()


# --- _is_under_path ---


def test_is_under_path_root():
    assert _is_under_path("https://example.com/anything", "/") is True


def test_is_under_path_empty():
    assert _is_under_path("https://example.com/anything", "") is True


def test_is_under_path_exact_match():
    assert _is_under_path("https://example.com/docs", "/docs") is True


def test_is_under_path_child():
    assert _is_under_path("https://example.com/docs/guide", "/docs") is True


def test_is_under_path_no_match():
    assert _is_under_path("https://example.com/blog/post", "/docs") is False


def test_is_under_path_partial_name():
    assert _is_under_path("https://example.com/docs-old/page", "/docs") is False


# --- _normalize_url ---


def test_normalize_url_strips_fragment():
    assert (
        _normalize_url("https://test.example.com/page#section") == "https://test.example.com/page"
    )


def test_normalize_url_no_fragment():
    assert _normalize_url("https://test.example.com/page") == "https://test.example.com/page"


# --- _extract_links ---


def test_extract_links_basic():
    html = '<a href="https://test.example.com/a">A</a><a href="https://test.example.com/b">B</a>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == ["https://test.example.com/a", "https://test.example.com/b"]


def test_extract_links_resolves_relative():
    html = '<a href="/docs/guide">Guide</a>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == ["https://test.example.com/docs/guide"]


def test_extract_links_filters_other_domain():
    html = '<a href="https://other.example.com/page">Other</a><a href="/local">Local</a>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == ["https://test.example.com/local"]


def test_extract_links_skips_non_http():
    html = '<a href="mailto:a@b.com">M</a><a href="javascript:void(0)">J</a>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == []


def test_extract_links_ignores_non_anchor_tags():
    html = '<div class="x"><a href="/page">P</a><span>text</span></div>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == ["https://test.example.com/page"]


def test_extract_links_skips_anchor_without_href():
    html = '<a class="btn">No href</a><a href="/ok">OK</a>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == ["https://test.example.com/ok"]


def test_extract_links_deduplicates():
    html = '<a href="/a">A</a><a href="/a#s1">A</a><a href="/a">A</a>'
    result = _extract_links(html, "https://test.example.com/")
    assert result == ["https://test.example.com/a"]


# --- _discover_urls_from_sitemap ---


SITEMAP_XML = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://test.example.com/</loc></url>
  <url><loc>https://test.example.com/docs</loc></url>
  <url><loc>https://test.example.com/about</loc></url>
</urlset>
"""


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_success(mock_urlopen):
    mock_urlopen.return_value.__enter__ = lambda s: BytesIO(SITEMAP_XML)
    mock_urlopen.return_value.__exit__ = lambda s, *a: None
    result = _discover_urls_from_sitemap("https://test.example.com/", 50)
    assert result == [
        "https://test.example.com/",
        "https://test.example.com/docs",
        "https://test.example.com/about",
    ]


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_not_found(mock_urlopen):
    mock_urlopen.side_effect = URLError("not found")
    result = _discover_urls_from_sitemap("https://test.example.com/", 50)
    assert result is None


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_invalid_xml(mock_urlopen):
    mock_urlopen.return_value.__enter__ = lambda s: BytesIO(b"not xml")
    mock_urlopen.return_value.__exit__ = lambda s, *a: None
    result = _discover_urls_from_sitemap("https://test.example.com/", 50)
    assert result is None


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_no_locs(mock_urlopen):
    xml = b'<?xml version="1.0"?><root><other>data</other></root>'
    mock_urlopen.return_value.__enter__ = lambda s: BytesIO(xml)
    mock_urlopen.return_value.__exit__ = lambda s, *a: None
    result = _discover_urls_from_sitemap("https://test.example.com/", 50)
    assert result is None


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_skips_null_loc_text(mock_urlopen):
    xml = b"""\
<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc></loc></url>
  <url><loc>https://test.example.com/real</loc></url>
</urlset>
"""
    mock_urlopen.return_value.__enter__ = lambda s: BytesIO(xml)
    mock_urlopen.return_value.__exit__ = lambda s, *a: None
    result = _discover_urls_from_sitemap("https://test.example.com/", 50)
    assert result == ["https://test.example.com/real"]


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_filters_by_path(mock_urlopen):
    xml = b"""\
<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://test.example.com/</loc></url>
  <url><loc>https://test.example.com/docs</loc></url>
  <url><loc>https://test.example.com/docs/guide</loc></url>
  <url><loc>https://test.example.com/blog</loc></url>
</urlset>
"""
    mock_urlopen.return_value.__enter__ = lambda s: BytesIO(xml)
    mock_urlopen.return_value.__exit__ = lambda s, *a: None
    result = _discover_urls_from_sitemap("https://test.example.com/docs", 50)
    assert result == [
        "https://test.example.com/docs",
        "https://test.example.com/docs/guide",
    ]


@patch("kod.pipeline.extract.urlopen")
def test_discover_sitemap_respects_max_pages(mock_urlopen):
    mock_urlopen.return_value.__enter__ = lambda s: BytesIO(SITEMAP_XML)
    mock_urlopen.return_value.__exit__ = lambda s, *a: None
    result = _discover_urls_from_sitemap("https://test.example.com/", 2)
    assert len(result) == 2


# --- _discover_urls_by_crawling ---


def _mock_urlopen_pages(pages):
    """Return a side_effect for urlopen that serves pages by URL."""

    def side_effect(url, timeout=None):
        if url in pages:
            resp = BytesIO(pages[url].encode())
            resp.close = lambda: None
            return resp
        raise URLError(f"not found: {url}")

    return side_effect


@patch("kod.pipeline.extract.urlopen")
def test_discover_crawl_basic(mock_urlopen):
    pages = {
        "https://test.example.com/": '<a href="/about">About</a>',
        "https://test.example.com/about": "<p>About page</p>",
    }
    mock_urlopen.side_effect = _mock_urlopen_pages(pages)
    result = _discover_urls_by_crawling("https://test.example.com/", 50)
    urls = [url for url, _ in result]
    assert "https://test.example.com/" in urls
    assert "https://test.example.com/about" in urls
    assert len(result) == 2


@patch("kod.pipeline.extract.urlopen")
def test_discover_crawl_filters_by_path(mock_urlopen):
    pages = {
        "https://test.example.com/docs": '<a href="/docs/guide">G</a><a href="/blog">B</a>',
        "https://test.example.com/docs/guide": "<p>Guide</p>",
        "https://test.example.com/blog": "<p>Blog</p>",
    }
    mock_urlopen.side_effect = _mock_urlopen_pages(pages)
    result = _discover_urls_by_crawling("https://test.example.com/docs", 50)
    urls = [url for url, _ in result]
    assert "https://test.example.com/docs" in urls
    assert "https://test.example.com/docs/guide" in urls
    assert "https://test.example.com/blog" not in urls


@patch("kod.pipeline.extract.urlopen")
def test_discover_crawl_respects_max_pages(mock_urlopen):
    pages = {
        "https://test.example.com/": '<a href="/a">A</a><a href="/b">B</a>',
        "https://test.example.com/a": "<p>A</p>",
        "https://test.example.com/b": "<p>B</p>",
    }
    mock_urlopen.side_effect = _mock_urlopen_pages(pages)
    result = _discover_urls_by_crawling("https://test.example.com/", 2)
    assert len(result) == 2


@patch("kod.pipeline.extract.urlopen")
def test_discover_crawl_handles_errors(mock_urlopen):
    pages = {
        "https://test.example.com/": '<a href="/good">G</a><a href="/bad">B</a>',
        "https://test.example.com/good": "<p>Good</p>",
    }
    mock_urlopen.side_effect = _mock_urlopen_pages(pages)
    result = _discover_urls_by_crawling("https://test.example.com/", 50)
    urls = [url for url, _ in result]
    assert "https://test.example.com/" in urls
    assert "https://test.example.com/good" in urls
    assert len(result) == 2


@patch("kod.pipeline.extract.urlopen")
def test_discover_crawl_deduplicates(mock_urlopen):
    pages = {
        "https://test.example.com/": '<a href="/a">A</a><a href="/b">B</a>',
        "https://test.example.com/a": '<a href="/b">B</a>',
        "https://test.example.com/b": '<a href="/">Home</a>',
    }
    mock_urlopen.side_effect = _mock_urlopen_pages(pages)
    result = _discover_urls_by_crawling("https://test.example.com/", 50)
    urls = [url for url, _ in result]
    assert len(result) == 3
    assert urls.count("https://test.example.com/b") == 1


# --- _extract_web_source ---


@patch("kod.pipeline.extract.partition_html")
@patch("kod.pipeline.extract._discover_urls_from_sitemap")
def test_extract_web_source_sitemap_path(mock_sitemap, mock_partition):
    mock_sitemap.return_value = [
        "https://example.com/docs",
        "https://example.com/docs/guide",
    ]
    mock_partition.return_value = [FakeElement("Content")]
    source = _web_source()

    docs = _extract_web_source(source)

    assert len(docs) == 2
    assert docs[0].file_path == "/docs"
    assert docs[1].file_path == "/docs/guide"
    assert all(d.source_name == "test-web" for d in docs)
    assert mock_partition.call_count == 2
    mock_partition.assert_any_call(url="https://example.com/docs")
    mock_partition.assert_any_call(url="https://example.com/docs/guide")


@patch("kod.pipeline.extract.partition_html")
@patch("kod.pipeline.extract._discover_urls_by_crawling")
@patch("kod.pipeline.extract._discover_urls_from_sitemap")
def test_extract_web_source_skips_sitemap_when_disabled(mock_sitemap, mock_crawl, mock_partition):
    mock_crawl.return_value = [("https://example.com/docs", "<p>Page</p>")]
    mock_partition.return_value = [FakeElement("Content")]
    source = _web_source(use_sitemap=False)

    docs = _extract_web_source(source)

    mock_sitemap.assert_not_called()
    assert len(docs) == 1


@patch("kod.pipeline.extract.partition_html")
@patch("kod.pipeline.extract._discover_urls_by_crawling")
@patch("kod.pipeline.extract._discover_urls_from_sitemap")
def test_extract_web_source_crawl_fallback(mock_sitemap, mock_crawl, mock_partition):
    mock_sitemap.return_value = None
    mock_crawl.return_value = [
        ("https://example.com/docs", "<p>Page 1</p>"),
        ("https://example.com/docs/about", "<p>Page 2</p>"),
    ]
    mock_partition.return_value = [FakeElement("Parsed")]
    source = _web_source()

    docs = _extract_web_source(source)

    assert len(docs) == 2
    assert docs[0].file_path == "/docs"
    assert docs[1].file_path == "/docs/about"
    mock_partition.assert_any_call(text="<p>Page 1</p>")
    mock_partition.assert_any_call(text="<p>Page 2</p>")


@patch("kod.pipeline.extract.partition_html")
@patch("kod.pipeline.extract._discover_urls_from_sitemap")
def test_extract_web_source_sets_file_path(mock_sitemap, mock_partition):
    mock_sitemap.return_value = ["https://example.com/"]
    mock_partition.return_value = [FakeElement("Root")]
    source = _web_source()

    docs = _extract_web_source(source)

    assert len(docs) == 1
    assert docs[0].file_path == "/"


@patch("kod.pipeline.extract.partition_html")
@patch("kod.pipeline.extract._discover_urls_by_crawling")
@patch("kod.pipeline.extract._discover_urls_from_sitemap")
def test_extract_web_source_crawl_skips_empty(mock_sitemap, mock_crawl, mock_partition):
    mock_sitemap.return_value = None
    mock_crawl.return_value = [
        ("https://example.com/a", "<p>Good</p>"),
        ("https://example.com/b", "<p></p>"),
    ]
    mock_partition.side_effect = [
        [FakeElement("Content")],
        [],
    ]

    docs = _extract_web_source(_web_source())

    assert len(docs) == 1
    assert docs[0].file_path == "/a"


@patch("kod.pipeline.extract.partition_html")
@patch("kod.pipeline.extract._discover_urls_from_sitemap")
def test_extract_web_source_skips_empty_pages(mock_sitemap, mock_partition):
    mock_sitemap.return_value = [
        "https://example.com/a",
        "https://example.com/b",
    ]
    mock_partition.side_effect = [
        [FakeElement("Content")],
        [],
    ]

    docs = _extract_web_source(_web_source())

    assert len(docs) == 1
    assert docs[0].file_path == "/a"


# --- _extract_git_source ---


@patch("kod.pipeline.extract.partition")
@patch("kod.pipeline.extract._clone_repo")
def test_extract_git_source(mock_clone, mock_partition, tmp_path):
    def create_files(url, target):
        target.mkdir(parents=True)
        (target / "doc.md").write_text("# Title")
        (target / "page.html").write_text("<h1>Hello</h1>")
        (target / "code.py").write_text("print()")

    mock_clone.side_effect = create_files
    mock_partition.return_value = [FakeElement("Parsed content")]

    source = _git_source()
    docs = _extract_git_source(source, tmp_path, {".md", ".html", ".htm"})

    assert len(docs) == 2
    assert mock_partition.call_count == 2
    paths = {d.file_path for d in docs}
    assert "doc.md" in paths
    assert "page.html" in paths
    assert all(d.source_name == "test-repo" for d in docs)
    assert all(d.metadata == {"product": "Test"} for d in docs)


@patch("kod.pipeline.extract.partition")
@patch("kod.pipeline.extract._clone_repo")
def test_extract_git_source_skips_empty(mock_clone, mock_partition, tmp_path):
    def create_files(url, target):
        target.mkdir(parents=True)
        (target / "empty.md").write_text("")

    mock_clone.side_effect = create_files
    mock_partition.return_value = []

    docs = _extract_git_source(_git_source(), tmp_path, {".md"})
    assert docs == []


# --- _extract_source ---


@patch("kod.pipeline.extract._extract_git_source")
def test_extract_source_git(mock_git, tmp_path):
    source = _git_source()
    extensions = {".md"}
    _extract_source(source, tmp_path, extensions)
    mock_git.assert_called_once_with(source, tmp_path, extensions)


@patch("kod.pipeline.extract._extract_web_source")
def test_extract_source_web(mock_web, tmp_path):
    source = _web_source()
    _extract_source(source, tmp_path, {".md"})
    mock_web.assert_called_once_with(source)


# --- run_extract ---


@patch("kod.pipeline.extract._extract_source")
def test_run_extract(mock_extract, tmp_path):
    mock_extract.return_value = [
        Document(
            content="Hello",
            source_name="test",
            source_url="https://example.com",
        )
    ]
    config = KodConfig(
        sources=[_web_source()],
        data_dir=tmp_path,
    )

    run_extract(config)

    output = tmp_path / "extracted" / "test-web.jsonl"
    assert output.exists()
    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1


@patch("kod.pipeline.extract._extract_source")
def test_run_extract_empty_source(mock_extract, tmp_path):
    mock_extract.return_value = []
    config = KodConfig(
        sources=[_web_source()],
        data_dir=tmp_path,
    )

    run_extract(config)

    output = tmp_path / "extracted" / "test-web.jsonl"
    assert output.exists()
    assert output.read_text() == ""


@patch("kod.pipeline.extract._extract_source")
def test_run_extract_continues_after_source_failure(mock_extract, tmp_path):
    good_doc = Document(
        content="Hello",
        source_name="good",
        source_url="https://example.com",
    )
    mock_extract.side_effect = [RuntimeError("clone failed"), [good_doc]]
    config = KodConfig(
        sources=[
            _web_source(name="bad-source"),
            _web_source(name="good-source"),
        ],
        data_dir=tmp_path,
    )

    run_extract(config)

    assert not (tmp_path / "extracted" / "bad-source.jsonl").exists()
    good_output = tmp_path / "extracted" / "good-source.jsonl"
    assert good_output.exists()
    assert good_output.read_text().strip() != ""
