"""Local SQLite FTS5 index over walked documents.

Why SQLite FTS5 first, vector embeddings later:
* zero infra to install — FTS5 ships with stdlib sqlite3
* good-enough precision for code + markdown corpora (BM25 ranking)
* fast: 100k documents indexed in <30s on commodity hardware
* trivially auditable — every match comes with a verbatim snippet

a4 adds an embedding column populated from a local model (sentence-
transformers or similar) for semantic recall on top of lexical FTS.
The schema is designed so embeddings slot in without a migration.

The index lives at ~/.hermes-memory/<project-hash>/index.db by default.
Per-project isolation means multiple projects don't pollute each other's
search space.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .walker import Document

# Bump this when schema changes; init() raises on mismatch so callers can
# rebuild rather than silently querying a stale shape.
SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    doc_type TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    -- Embedding column reserved for a4. NULL until populated. Stored as
    -- raw bytes (float32 little-endian, 384-dim by convention).
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_mtime ON documents(mtime);

-- FTS5 virtual table for lexical search. content='' means we store the
-- text inside FTS itself (external-content tables save space but make
-- updates more brittle; for a3 we prefer simplicity).
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    file_path UNINDEXED,
    doc_type UNINDEXED,
    tokenize = 'porter unicode61'
);
"""


@dataclass(frozen=True)
class SearchHit:
    """One result from index.search()."""

    file_path: str
    doc_type: str
    snippet: str
    rank: float


def default_index_dir(project_root: Path) -> Path:
    """Per-project index dir under ~/.hermes-memory/. Hash the absolute
    path so two different checkouts of the same repo get separate indexes
    rather than stomping each other."""
    abs_root = str(project_root.expanduser().resolve())
    digest = hashlib.sha256(abs_root.encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".hermes-memory" / digest


class Index:
    """Thin wrapper around a SQLite connection with FTS5 + docs schema.

    Use as a context manager so the connection always closes:

        with Index.open(project_root) as ix:
            ix.add(doc)
            hits = ix.search("query")
    """

    def __init__(self, db_path: Path, conn: sqlite3.Connection) -> None:
        self.db_path = db_path
        self.conn = conn

    @classmethod
    def open(cls, project_root: Path) -> Index:
        """Open (or create) the index for a project root."""
        index_dir = default_index_dir(project_root)
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "index.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        # Record schema version. If it mismatches an existing DB, raise so
        # the caller can rebuild rather than corrupting silently.
        existing = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('project_root', ?)",
                (str(project_root.expanduser().resolve()),),
            )
        elif int(existing["value"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Index at {db_path} has schema_version={existing['value']!r}; "
                f"current is {SCHEMA_VERSION}. Delete and rebuild "
                f"(rm -rf {index_dir})."
            )
        conn.commit()
        return cls(db_path, conn)

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ── ingest ────────────────────────────────────────────────────────

    def add(self, doc: Document) -> None:
        """Index a single document. Replaces any existing entry for the
        same file_path so re-indexing is idempotent."""
        cur = self.conn.execute(
            "SELECT id FROM documents WHERE file_path = ?",
            (str(doc.file_path),),
        )
        existing = cur.fetchone()
        if existing is not None:
            self.conn.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))
            self.conn.execute(
                "DELETE FROM documents_fts WHERE file_path = ?",
                (str(doc.file_path),),
            )

        self.conn.execute(
            """INSERT INTO documents (file_path, doc_type, mtime, size)
               VALUES (?, ?, ?, ?)""",
            (str(doc.file_path), doc.doc_type, doc.mtime, doc.size),
        )
        self.conn.execute(
            """INSERT INTO documents_fts (content, file_path, doc_type)
               VALUES (?, ?, ?)""",
            (doc.content, str(doc.file_path), doc.doc_type),
        )

    def add_many(self, docs) -> int:
        """Index an iterable of Documents. Commits once at the end for
        throughput. Returns the number of documents added."""
        count = 0
        for doc in docs:
            self.add(doc)
            count += 1
        self.conn.commit()
        return count

    # ── query ─────────────────────────────────────────────────────────

    def doc_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]

    def doc_types(self) -> dict[str, int]:
        """Return {doc_type: count} for the indexed corpus."""
        rows = self.conn.execute(
            "SELECT doc_type, COUNT(*) AS n FROM documents GROUP BY doc_type"
        ).fetchall()
        return {row["doc_type"]: row["n"] for row in rows}

    def search(
        self,
        query: str,
        *,
        scope: str = "all",
        limit: int = 10,
    ) -> list[SearchHit]:
        """Run an FTS5 search and return ranked hits.

        ``scope`` filters by doc_type. 'all' = no filter; the rest match
        the scopes declared in schemas.SEARCH_MEMORY_SCHEMA.
        """
        if not query.strip():
            return []

        # FTS5 query syntax: wrap user query in quotes so punctuation
        # doesn't blow up the parser. Power users can pass raw FTS5 syntax
        # by setting it through the index API directly; the MCP surface
        # always wraps.
        fts_query = '"' + query.replace('"', '""') + '"'

        if scope == "all":
            rows = self.conn.execute(
                """SELECT file_path, doc_type,
                          snippet(documents_fts, 0, '<<<', '>>>', '...', 16) AS snip,
                          bm25(documents_fts) AS rank
                   FROM documents_fts
                   WHERE documents_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT file_path, doc_type,
                          snippet(documents_fts, 0, '<<<', '>>>', '...', 16) AS snip,
                          bm25(documents_fts) AS rank
                   FROM documents_fts
                   WHERE documents_fts MATCH ? AND doc_type = ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, scope, limit),
            ).fetchall()

        return [
            SearchHit(
                file_path=row["file_path"],
                doc_type=row["doc_type"],
                snippet=row["snip"],
                rank=float(row["rank"]),
            )
            for row in rows
        ]
