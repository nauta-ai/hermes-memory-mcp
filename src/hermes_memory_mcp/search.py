"""Hybrid lexical + semantic search via Reciprocal Rank Fusion.

a5 design notes:
* The two retrievers ([FTS5 BM25][bm25] and brute-force cosine over stored
  embeddings) measure different things — BM25 surfaces exact-term hits;
  embeddings surface paraphrase / topical hits. Blending both consistently
  outperforms either alone for natural-language queries over mixed-format
  corpora (markdown notes + code + ADRs + commit logs).
* We use [Reciprocal Rank Fusion (RRF)][rrf] rather than scaled-score
  blending because the two retrievers' raw scores aren't on the same
  scale (BM25 is unbounded negative, cosine is bounded [-1, 1]) and
  normalizing them well is fiddly. RRF only cares about *rank* and is
  parameter-light (the constant ``k`` defaults to 60, established
  in the original paper).
* When ``embedder`` is None the function degrades cleanly to FTS5-only —
  this keeps the a3/a4 code path intact for users who don't want to
  install the optional embeddings extra.

[bm25]: https://en.wikipedia.org/wiki/Okapi_BM25
[rrf]: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

from __future__ import annotations

from .embedder import Embedder
from .index import Index, SearchHit

# RRF constant from the original 2009 paper. Larger values give earlier
# ranks proportionally less dominance; the paper's empirical sweet spot
# was k=60 and we have no reason to deviate.
RRF_K = 60


def hybrid_search(
    index: Index,
    query: str,
    *,
    scope: str = "all",
    limit: int = 10,
    embedder: Embedder | None = None,
    candidate_pool: int = 40,
    raw_fts: bool = False,
) -> list[SearchHit]:
    """Return ranked SearchHits, blending FTS + vector search via RRF.

    Args:
        index: the SQLite-backed index produced by :class:`Index.open`.
        query: natural-language search string.
        scope: doc_type filter forwarded to both retrievers.
        limit: number of results to return after fusion.
        embedder: if provided, vector search runs and results fuse via
            RRF. If ``None``, falls back to pure FTS5 (drop-in compatible
            with ``index.search()`` so a3-era callers see no change).
        candidate_pool: how many candidates each retriever produces
            before fusion. RRF benefits from a wider pool than ``limit``
            because the final top-N is the *blend* of two top-N lists,
            so 4x ``limit`` per retriever is the rule of thumb.
        raw_fts: passed through to FTS5 — for advanced operator syntax.

    The fusion runs entirely in Python and is dominated by SQLite I/O;
    the per-call overhead beyond the underlying searches is ~1 ms for
    a candidate pool of 40 each.
    """
    if embedder is None:
        # Pure FTS5 path — keeps a3/a4 behavior bit-identical when
        # embeddings aren't configured. Callers shouldn't have to know
        # whether the embedder is loaded.
        return index.search(query, scope=scope, limit=limit, raw_fts=raw_fts)

    fts_hits = index.search(query, scope=scope, limit=candidate_pool, raw_fts=raw_fts)
    query_vec = embedder.embed_one(query)
    vec_hits = index.vector_search(query_vec, scope=scope, limit=candidate_pool)

    return _rrf_fuse(fts_hits, vec_hits, limit=limit)


def _rrf_fuse(
    fts_hits: list[SearchHit],
    vec_hits: list[SearchHit],
    *,
    limit: int,
    k: int = RRF_K,
) -> list[SearchHit]:
    """Reciprocal Rank Fusion of two ranked SearchHit lists.

    For each file_path, sum ``1 / (k + rank_i)`` across the two lists
    (rank counted from 0). Documents that appear in *both* lists
    naturally bubble up.

    Returns SearchHits with ``rank`` set to the fused RRF score (higher
    is better) and the snippet taken from whichever retriever had the
    higher individual rank for that document.
    """
    scores: dict[str, float] = {}
    # Track the best (lowest-rank, hence higher-quality) hit per
    # file_path so we can return a sensible snippet. FTS5's snippet
    # markers are nicer than the vector path's leading-character slice,
    # so when both surfaces produce a hit we prefer the FTS snippet.
    best_hit: dict[str, SearchHit] = {}

    for ranking, hits in enumerate((fts_hits, vec_hits)):
        for rank, hit in enumerate(hits):
            scores[hit.file_path] = scores.get(hit.file_path, 0.0) + 1.0 / (k + rank)
            # First retriever (FTS, ranking == 0) wins ties for snippet
            # because its snippets carry << >> highlighting.
            if hit.file_path not in best_hit or (ranking == 0 and rank < 5):
                best_hit[hit.file_path] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [
        SearchHit(
            file_path=fp,
            doc_type=best_hit[fp].doc_type,
            snippet=best_hit[fp].snippet,
            rank=score,
        )
        for fp, score in ranked
    ]
