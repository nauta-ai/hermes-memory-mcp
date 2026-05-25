"""Tests for the a5 embedding methods on :class:`Index`.

Split out from ``test_index.py`` so the FTS-only tests there stay
runnable without the optional fastembed extra.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_memory_mcp import index as index_mod
from hermes_memory_mcp.index import Index
from hermes_memory_mcp.walker import Document

fastembed = pytest.importorskip("fastembed")
from hermes_memory_mcp.embedder import Embedder, reset_default  # noqa: E402


@pytest.fixture
def tmp_index(tmp_path: Path):
    with patch.object(index_mod, "default_index_dir", lambda root: tmp_path / "_index"):
        ix = Index.open(tmp_path)
    yield ix
    ix.close()


def _doc(path: str, content: str, doc_type: str = "markdown") -> Document:
    return Document(
        file_path=Path(path),
        doc_type=doc_type,
        content=content,
        mtime=1.0,
        size=len(content),
    )


@pytest.fixture(autouse=True)
def _reset_embedder():
    reset_default()
    yield
    reset_default()


def test_embedding_coverage_starts_at_zero(tmp_index):
    tmp_index.add_many([_doc("/a", "x"), _doc("/b", "y")])
    embedded, total = tmp_index.embedding_coverage()
    assert (embedded, total) == (0, 2)


def test_embed_all_pending_covers_all_docs(tmp_index):
    docs = [_doc(f"/d{i}", f"content {i}") for i in range(5)]
    tmp_index.add_many(docs)
    emb = Embedder()
    added = tmp_index.embed_all_pending(emb)
    assert added == 5
    embedded, total = tmp_index.embedding_coverage()
    assert (embedded, total) == (5, 5)


def test_embed_all_pending_is_idempotent(tmp_index):
    """Running twice in a row should be a no-op the second time —
    important for the catch-up cadence on long-running daemons that
    re-call ``embed`` after every walk."""
    tmp_index.add_many([_doc("/a", "x"), _doc("/b", "y")])
    emb = Embedder()
    first = tmp_index.embed_all_pending(emb)
    second = tmp_index.embed_all_pending(emb)
    assert first == 2
    assert second == 0


def test_embed_all_pending_picks_up_new_docs(tmp_index):
    tmp_index.add_many([_doc("/a", "x")])
    emb = Embedder()
    tmp_index.embed_all_pending(emb)
    # New doc added after first embed run
    tmp_index.add(_doc("/b", "y"))
    tmp_index.conn.commit()
    added = tmp_index.embed_all_pending(emb)
    assert added == 1


def test_vector_search_finds_semantic_match(tmp_index):
    tmp_index.add_many(
        [
            _doc("/auth.md", "User authentication via JWT tokens and bcrypt."),
            _doc("/db.md", "Postgres orders table schema."),
            _doc("/ui.md", "React components for the dashboard."),
        ]
    )
    emb = Embedder()
    tmp_index.embed_all_pending(emb)
    query_vec = emb.embed_one("how do users sign in")
    hits = tmp_index.vector_search(query_vec, limit=3)
    # Top hit should be the auth doc (semantic match), not db or ui.
    assert len(hits) > 0
    assert "auth" in hits[0].file_path


def test_vector_search_respects_scope_filter(tmp_index):
    tmp_index.add_many(
        [
            _doc("/adr-007.md", "Decided on JWT for auth.", doc_type="adr"),
            _doc("/auth.md", "JWT auth implementation.", doc_type="markdown"),
        ]
    )
    emb = Embedder()
    tmp_index.embed_all_pending(emb)
    q = emb.embed_one("auth decision")
    hits = tmp_index.vector_search(q, scope="adr", limit=10)
    assert all(h.doc_type == "adr" for h in hits)


def test_vector_search_skips_documents_without_embedding(tmp_index):
    """Documents added after the last embed pass have ``embedding IS NULL``
    and must NOT appear in vector_search results (cosine of NULL would
    crash). They'll show up on the next ``embed_all_pending`` run."""
    tmp_index.add_many([_doc("/a", "first")])
    emb = Embedder()
    tmp_index.embed_all_pending(emb)
    tmp_index.add(_doc("/b", "second"))  # not yet embedded
    tmp_index.conn.commit()
    q = emb.embed_one("second")
    hits = tmp_index.vector_search(q, limit=10)
    # Only the embedded doc should appear — even though /b semantically
    # matches the query better.
    assert all(h.file_path == "/a" for h in hits)


def test_update_embedding_no_op_for_unknown_path(tmp_index):
    """Updating embeddings for a path that isn't in the docs table
    silently does nothing — derivative-data write should never create
    phantom rows."""
    fake_vec = b"\x00" * (384 * 4)
    tmp_index.update_embedding("/does/not/exist", fake_vec)
    # No row was created
    cur = tmp_index.conn.execute(
        "SELECT COUNT(*) AS n FROM documents WHERE file_path = '/does/not/exist'"
    ).fetchone()
    assert cur["n"] == 0
