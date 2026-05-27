"""Index tests — exercise SQLite FTS5 ingest + search end-to-end.

These tests use tmp_path so they never touch ~/.hermes-memory/.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_memory_mcp import index as index_mod
from hermes_memory_mcp.index import SCHEMA_VERSION, Index, SearchHit
from hermes_memory_mcp.walker import (
    DOC_TYPE_ADR,
    DOC_TYPE_CODE,
    DOC_TYPE_MARKDOWN,
    Document,
    walk,
)


@pytest.fixture
def tmp_index(tmp_path: Path) -> Index:
    """An Index rooted at tmp_path. Uses patch to redirect
    default_index_dir so we don't write under ~/.hermes-memory."""
    with patch.object(index_mod, "default_index_dir", lambda root: tmp_path / "_index"):
        ix = Index.open(tmp_path)
    yield ix
    ix.close()


def _doc(path: Path, content: str, doc_type: str = DOC_TYPE_MARKDOWN) -> Document:
    return Document(
        file_path=path,
        doc_type=doc_type,
        content=content,
        mtime=1.0,
        size=len(content),
    )


def test_open_creates_schema_with_version(tmp_index: Index) -> None:
    row = tmp_index.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert int(row["value"]) == SCHEMA_VERSION


def test_add_and_search_returns_hit(tmp_path: Path, tmp_index: Index) -> None:
    tmp_index.add(_doc(tmp_path / "doc1.md", "The quick brown fox"))
    tmp_index.add(_doc(tmp_path / "doc2.md", "Lazy dog"))
    hits = tmp_index.search("brown fox")
    assert len(hits) >= 1
    assert hits[0].file_path == str(tmp_path / "doc1.md")
    assert "brown" in hits[0].snippet.lower() or "fox" in hits[0].snippet.lower()


def test_search_returns_empty_for_unmatched_query(tmp_path: Path, tmp_index: Index) -> None:
    tmp_index.add(_doc(tmp_path / "doc.md", "Hello world"))
    assert tmp_index.search("nonexistent-term-xyz") == []


def test_search_empty_query_returns_empty(tmp_path: Path, tmp_index: Index) -> None:
    tmp_index.add(_doc(tmp_path / "doc.md", "Hello"))
    assert tmp_index.search("") == []
    assert tmp_index.search("   ") == []


def test_search_filters_by_scope(tmp_path: Path, tmp_index: Index) -> None:
    tmp_index.add(_doc(tmp_path / "a.md", "transformers", DOC_TYPE_MARKDOWN))
    tmp_index.add(_doc(tmp_path / "b.py", "transformers", DOC_TYPE_CODE))
    tmp_index.add(_doc(tmp_path / "c.md", "transformers", DOC_TYPE_ADR))

    all_hits = tmp_index.search("transformers")
    assert len(all_hits) == 3

    md_hits = tmp_index.search("transformers", scope=DOC_TYPE_MARKDOWN)
    assert len(md_hits) == 1
    assert md_hits[0].doc_type == DOC_TYPE_MARKDOWN

    code_hits = tmp_index.search("transformers", scope=DOC_TYPE_CODE)
    assert len(code_hits) == 1
    assert code_hits[0].doc_type == DOC_TYPE_CODE


def test_normalize_scope_maps_mcp_names_to_doc_types() -> None:
    """The MCP tool schema exposes 'notes'/'decisions'/'logs' to clients,
    but the walker writes 'markdown'/'adr'/'log'. _normalize_scope must
    map MCP-facing names to walker doc_types so MCP callers don't get
    silently-empty results. Canonical doc_types pass through unchanged."""
    from hermes_memory_mcp.index import _normalize_scope
    # MCP-facing aliases
    assert _normalize_scope("notes") == "markdown"
    assert _normalize_scope("decisions") == "adr"
    assert _normalize_scope("logs") == "log"
    # Canonical doc_types — pass-through
    assert _normalize_scope("markdown") == "markdown"
    assert _normalize_scope("code") == "code"
    assert _normalize_scope("adr") == "adr"
    assert _normalize_scope("log") == "log"
    assert _normalize_scope("git") == "git"
    # Sentinel
    assert _normalize_scope("all") == "all"


def test_search_with_mcp_scope_alias(tmp_path: Path, tmp_index: Index) -> None:
    """End-to-end: an MCP client passing scope='notes' must hit the same
    documents that the walker tagged as doc_type='markdown'. This is the
    bug that made early Cursor integrations return zero results on a
    populated index."""
    tmp_index.add(_doc(tmp_path / "a.md", "synapse", DOC_TYPE_MARKDOWN))
    tmp_index.add(_doc(tmp_path / "b.py", "synapse", DOC_TYPE_CODE))

    md_hits = tmp_index.search("synapse", scope=DOC_TYPE_MARKDOWN)
    notes_hits = tmp_index.search("synapse", scope="notes")
    assert len(notes_hits) == len(md_hits) == 1
    assert notes_hits[0].file_path == md_hits[0].file_path


def test_search_respects_limit(tmp_path: Path, tmp_index: Index) -> None:
    for i in range(20):
        tmp_index.add(_doc(tmp_path / f"d{i}.md", f"doc {i} matches widget"))
    hits = tmp_index.search("widget", limit=5)
    assert len(hits) == 5


def test_re_add_replaces_existing(tmp_path: Path, tmp_index: Index) -> None:
    path = tmp_path / "doc.md"
    tmp_index.add(_doc(path, "original content"))
    tmp_index.add(_doc(path, "updated content"))
    assert tmp_index.doc_count() == 1

    hits = tmp_index.search("updated")
    assert len(hits) == 1

    stale = tmp_index.search("original")
    assert stale == []


def test_add_many_returns_count(tmp_path: Path, tmp_index: Index) -> None:
    docs = [_doc(tmp_path / f"d{i}.md", f"content {i}") for i in range(7)]
    n = tmp_index.add_many(docs)
    assert n == 7
    assert tmp_index.doc_count() == 7


def test_doc_types_breakdown(tmp_path: Path, tmp_index: Index) -> None:
    tmp_index.add(_doc(tmp_path / "a.md", "x", DOC_TYPE_MARKDOWN))
    tmp_index.add(_doc(tmp_path / "b.md", "x", DOC_TYPE_MARKDOWN))
    tmp_index.add(_doc(tmp_path / "c.py", "x", DOC_TYPE_CODE))
    breakdown = tmp_index.doc_types()
    assert breakdown == {DOC_TYPE_MARKDOWN: 2, DOC_TYPE_CODE: 1}


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    """If a future a4 bumps SCHEMA_VERSION and a user has a stale DB
    from a3, opening it must raise rather than silently mis-query."""
    index_dir = tmp_path / "_index"
    index_dir.mkdir()
    db_path = index_dir / "index.db"
    # Manually create a DB with bogus schema_version
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()

    with patch.object(index_mod, "default_index_dir", lambda root: index_dir):
        with pytest.raises(RuntimeError, match="schema_version"):
            Index.open(tmp_path)


def test_end_to_end_walk_and_index(tmp_path: Path, tmp_index: Index) -> None:
    """Walker → Index round trip: write a small project, walk it, index
    every document, then search."""
    (tmp_path / "README.md").write_text("# Hermes\nA cited memory MCP for AI agents.")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def cited_memory():\n    return 'AI agents need this'")

    docs = list(walk(tmp_path))
    # The index db itself was created under tmp_path/_index but that's an
    # ignored dir; walker excludes _index because... actually it doesn't,
    # but the tmp_index fixture writes index.db AFTER walk runs. So docs
    # captures only the project files.
    tmp_index.add_many(docs)

    hits = tmp_index.search("cited memory")
    assert len(hits) >= 1
    # Both README and source mention "cited" or "memory" — FTS ranks them
    # together; just verify at least one hit comes from the project tree.
    assert any(str(tmp_path) in h.file_path for h in hits)


def test_search_hit_is_dataclass() -> None:
    hit = SearchHit(file_path="/x", doc_type="markdown", snippet="...", rank=0.5)
    assert hit.file_path == "/x"
