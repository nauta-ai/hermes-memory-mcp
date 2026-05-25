"""Tests for hybrid_search (RRF blending of FTS + vector retrievers).

These exercise the *fusion logic* rather than recall quality — measuring
recall meaningfully needs the Amnesia Test sample corpus (deferred to
a5.1). Here we verify:

* ``embedder=None`` degrades to pure FTS5 byte-for-byte
* Both retrievers contribute candidates and RRF blends them
* Documents in both lists score higher than documents in only one
* The scope filter propagates to both retrievers
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_memory_mcp.index import Index, SearchHit
from hermes_memory_mcp.search import _rrf_fuse, hybrid_search
from hermes_memory_mcp.walker import Document

fastembed = pytest.importorskip("fastembed")
from hermes_memory_mcp.embedder import Embedder, reset_default  # noqa: E402


@pytest.fixture
def small_corpus_index(tmp_path):
    """A handful of documents covering distinct topics so retrieval
    has a meaningful signal."""
    docs = [
        Document(
            file_path=Path("/x/auth.md"),
            doc_type="markdown",
            mtime=0.0,
            size=100,
            content="User authentication via JWT tokens. Login flow uses bcrypt.",
        ),
        Document(
            file_path=Path("/x/jwt-spec.md"),
            doc_type="markdown",
            mtime=0.0,
            size=100,
            content="Token validation logic with refresh token rotation.",
        ),
        Document(
            file_path=Path("/x/auth-adr.md"),
            doc_type="adr",
            mtime=0.0,
            size=100,
            content="ADR-007: chose JWT over session cookies for stateless auth.",
        ),
        Document(
            file_path=Path("/x/db.md"),
            doc_type="markdown",
            mtime=0.0,
            size=100,
            content="Database schema for orders table. Uses Postgres.",
        ),
        Document(
            file_path=Path("/x/order-service.md"),
            doc_type="markdown",
            mtime=0.0,
            size=100,
            content="Order placement microservice. Idempotent.",
        ),
    ]
    fake_root = tmp_path / "fakeproject"
    fake_root.mkdir()
    with Index.open(fake_root) as ix:
        ix.add_many(docs)
        yield ix


@pytest.fixture
def embedded_index(small_corpus_index):
    """An index where every document has its embedding populated."""
    reset_default()
    emb = Embedder()
    small_corpus_index.embed_all_pending(emb)
    yield small_corpus_index, emb
    reset_default()


def test_hybrid_falls_back_to_fts_when_embedder_is_none(small_corpus_index):
    """The legacy a3/a4 path must keep working bit-for-bit when no
    embedder is wired in. We compare against ``index.search()`` directly."""
    fts_only = small_corpus_index.search("authentication", limit=5)
    via_hybrid = hybrid_search(small_corpus_index, "authentication", embedder=None, limit=5)
    # Same ranking, same files, same ranks.
    assert [(h.file_path, h.rank) for h in fts_only] == [(h.file_path, h.rank) for h in via_hybrid]


def test_hybrid_returns_results_for_natural_language_query(embedded_index):
    """A semantic query that has no exact-term match should still find
    something via the vector retriever. With pure FTS this would be
    empty for queries like 'who is signing in'."""
    ix, emb = embedded_index
    # Note: "JWT" appears in docs, but a vague query without it should
    # still surface auth docs via vectors.
    hits = hybrid_search(ix, "how does login identification work", embedder=emb, limit=3)
    assert len(hits) > 0


def test_hybrid_scope_filter_applies_to_both_retrievers(embedded_index):
    ix, emb = embedded_index
    hits = hybrid_search(ix, "auth", scope="adr", embedder=emb, limit=10)
    # Only the auth-adr doc has doc_type == 'adr'
    assert all(h.doc_type == "adr" for h in hits)
    assert any("auth-adr" in h.file_path for h in hits)


def test_hybrid_limit_caps_returned_results(embedded_index):
    ix, emb = embedded_index
    hits = hybrid_search(ix, "token", embedder=emb, limit=2)
    assert len(hits) <= 2


def test_rrf_fuse_boosts_documents_in_both_lists():
    """If a doc shows up in both retrievers, it should rank above docs
    that show up in only one."""
    fts = [
        SearchHit(file_path="/a", doc_type="md", snippet="...", rank=-3.0),
        SearchHit(file_path="/b", doc_type="md", snippet="...", rank=-2.0),
    ]
    vec = [
        SearchHit(file_path="/c", doc_type="md", snippet="...", rank=0.9),
        SearchHit(file_path="/a", doc_type="md", snippet="...", rank=0.8),  # also in fts
    ]
    fused = _rrf_fuse(fts, vec, limit=10)
    paths = [h.file_path for h in fused]
    # /a appears in both — must rank above /b (FTS-only rank 1) and
    # above /c (vector-only rank 0).
    a_rank = paths.index("/a")
    b_rank = paths.index("/b")
    c_rank = paths.index("/c")
    assert a_rank < b_rank
    assert a_rank < c_rank


def test_rrf_fuse_uses_fts_snippet_when_both_match():
    """FTS5 snippets carry << >> highlighting and are more useful to
    LLMs than the vector path's leading slice. The fuser should prefer
    them when both retrievers hit the same file."""
    fts = [
        SearchHit(file_path="/x", doc_type="md", snippet="<<auth>> token", rank=-1.0),
    ]
    vec = [
        SearchHit(file_path="/x", doc_type="md", snippet="auth token...", rank=0.9),
    ]
    fused = _rrf_fuse(fts, vec, limit=10)
    assert len(fused) == 1
    assert "<<auth>>" in fused[0].snippet


def test_rrf_fuse_empty_inputs_returns_empty():
    assert _rrf_fuse([], [], limit=5) == []


def test_rrf_fuse_single_retriever_still_works():
    """If one retriever has no results (e.g. FTS phrase match fails on
    a fuzzy query), the other should still surface its hits."""
    fts: list[SearchHit] = []
    vec = [
        SearchHit(file_path="/x", doc_type="md", snippet="x", rank=0.9),
        SearchHit(file_path="/y", doc_type="md", snippet="y", rank=0.8),
    ]
    fused = _rrf_fuse(fts, vec, limit=5)
    assert [h.file_path for h in fused] == ["/x", "/y"]
