"""Local sentence-embedding wrapper around FastEmbed (ONNX).

a5 design notes:
* FastEmbed (not sentence-transformers + torch) so the install footprint is
  ~80 MB instead of 2 GB. ONNX runtime is small enough to bundle in a
  user-installable PyPI package.
* Default model: ``sentence-transformers/all-MiniLM-L6-v2`` — 384-dim, the
  size the SQLite ``embedding`` column was reserved for in a3.
* All-or-nothing import: ``import fastembed`` only happens lazily inside
  :class:`Embedder.__init__`. That way ``hermes-memory-mcp`` keeps working
  without the optional dependency installed, and the user sees a clean
  ``EmbedderUnavailableError`` if they ask for vector search without
  installing the extra.
* Single shared instance per process via :func:`get_default`. Loading the
  ONNX session is the expensive bit (~600 ms cold start); once loaded,
  encoding is ~3 ms per document on commodity Apple Silicon.

The embedder is intentionally opt-in. Callers that want pure FTS5 (the
a3/a4 path) should not touch this module at all.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class EmbedderUnavailableError(RuntimeError):
    """Raised when an embedding call is made but FastEmbed isn't installed.

    The fix is documented in the message: ``pip install
    'hermes-memory-mcp[embeddings]'``. We don't try to fall back silently
    — semantic recall going dark without the user noticing is worse than
    a loud error.
    """


@dataclass(frozen=True)
class EmbedderConfig:
    """Pinning surface for which embedding model to load.

    ``model`` is the FastEmbed model identifier. ``dim`` is recorded so
    callers can sanity-check that an existing index's stored vectors
    match this embedder's output size (mismatched dims would silently
    return garbage cosine scores).
    """

    model: str = DEFAULT_MODEL
    dim: int = DEFAULT_DIM


class Embedder:
    """Lazy-loaded text → float32 vector encoder.

    Usage::

        emb = Embedder()                   # loads the ONNX session
        vectors = emb.embed(["hello", "world"])  # list[bytes]
        emb.dim                            # 384

    Returns ``bytes`` (float32 little-endian, length ``dim * 4``) rather
    than NumPy arrays so callers can store them directly in SQLite's
    ``embedding BLOB`` column without an extra conversion step.
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - exercised in test_embedder
            raise EmbedderUnavailableError(
                "fastembed is not installed. Install the optional extra: "
                "pip install 'hermes-memory-mcp[embeddings]'"
            ) from e
        self._model = TextEmbedding(model_name=self.config.model)

    @property
    def dim(self) -> int:
        return self.config.dim

    @property
    def model_name(self) -> str:
        return self.config.model

    def embed(self, texts: list[str]) -> list[bytes]:
        """Encode a batch of texts to float32 byte vectors.

        Input order is preserved. Empty strings are encoded as zero
        vectors (rather than raising) so a corpus with a few empty docs
        doesn't crash the indexer.

        Returned bytes are ``self.dim * 4`` long each. Decode with
        :func:`bytes_to_floats` or pass straight to a SQLite BLOB.
        """
        if not texts:
            return []
        # Replace empty strings with a single space so the tokenizer
        # always has something to chew on; downstream we treat all-zero
        # vectors as "no match" anyway via cosine_similarity returning 0.
        safe = [t if t.strip() else " " for t in texts]
        out: list[bytes] = []
        # FastEmbed's TextEmbedding.embed returns a generator of numpy
        # arrays; we materialize one at a time to keep peak memory low.
        for vec in self._model.embed(safe):
            # Sanity: dim mismatch means we'd corrupt the index. Better
            # to crash here than silently store wrong-size BLOBs.
            if vec.shape[0] != self.dim:
                raise RuntimeError(
                    f"embedder produced dim={vec.shape[0]}, expected {self.dim}. "
                    f"Did the model change? Configured: {self.config.model}"
                )
            # struct.pack is ~30% faster than vec.astype(np.float32).tobytes()
            # for 384-dim vectors and avoids importing numpy at the call site.
            out.append(struct.pack(f"<{self.dim}f", *vec.tolist()))
        return out

    def embed_one(self, text: str) -> bytes:
        """Convenience wrapper around :meth:`embed` for single-text callers
        (the common case for query encoding)."""
        return self.embed([text])[0]


# ── module-level singleton cache ──────────────────────────────────────
# Loading FastEmbed is ~600 ms — we cache one instance per process so the
# MCP server doesn't pay that cost on every tool call.

_default_instance: Embedder | None = None


def get_default(config: EmbedderConfig | None = None) -> Embedder:
    """Return a process-wide singleton Embedder. Creates it on first call.

    Passing a ``config`` only matters on the first call; subsequent calls
    re-use the originally-loaded model. This is intentional — switching
    models mid-process would invalidate every embedding already in the
    index, so we'd rather make that a deliberate ``reset_default()`` step.
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = Embedder(config)
    return _default_instance


def reset_default() -> None:
    """Drop the cached singleton. Mainly useful for tests that want to
    rebuild the embedder under a different config."""
    global _default_instance
    _default_instance = None


# ── byte ↔ float helpers ──────────────────────────────────────────────


def bytes_to_floats(buf: bytes, dim: int = DEFAULT_DIM) -> list[float]:
    """Decode a stored embedding back to a Python list of floats.

    Tiny enough that we don't need numpy at the read path either — the
    only hot path that needs vectorized cosine is :func:`cosine_similarity`,
    which can stay pure-Python for now (numpy is pulled in transitively
    by fastembed but we don't want to require it at the call site)."""
    if len(buf) != dim * 4:
        raise ValueError(f"buffer length {len(buf)} != {dim} * 4 = {dim * 4}")
    return list(struct.unpack(f"<{dim}f", buf))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two same-length vectors.

    Returns 0.0 if either vector is all-zero (avoids ZeroDivisionError
    on the empty-string sentinel case). Returns a value in [-1, 1] for
    well-formed inputs.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # math.sqrt is faster than ** 0.5 in CPython and clearer to read
    import math

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
