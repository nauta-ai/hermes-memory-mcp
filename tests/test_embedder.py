"""Tests for the embedder module.

These tests load FastEmbed lazily and download the all-MiniLM-L6-v2 ONNX
weights on first run (~80 MB, cached at ~/.cache/fastembed). Subsequent
runs are offline. Tests are skipped wholesale when fastembed isn't
installed so the suite stays green for users who don't want the extra.
"""

from __future__ import annotations

import pytest

from hermes_memory_mcp.embedder import (
    DEFAULT_DIM,
    EmbedderUnavailableError,
    bytes_to_floats,
    cosine_similarity,
)

# Module-wide skip if fastembed isn't available. Avoids loading the
# (large) ONNX model when running ``pytest`` on a slim CI machine.
fastembed = pytest.importorskip("fastembed")

from hermes_memory_mcp.embedder import (  # noqa: E402
    Embedder,
    EmbedderConfig,
    get_default,
    reset_default,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test gets a fresh singleton state so cross-test
    config bleed doesn't hide bugs."""
    reset_default()
    yield
    reset_default()


def test_embedder_default_config_dim_matches_minilm():
    emb = Embedder()
    assert emb.dim == DEFAULT_DIM == 384
    assert "MiniLM" in emb.model_name


def test_embed_returns_correct_byte_length():
    emb = Embedder()
    vec = emb.embed_one("hello world")
    # 384 floats * 4 bytes each
    assert len(vec) == DEFAULT_DIM * 4


def test_embed_batch_preserves_order():
    emb = Embedder()
    texts = ["cat", "dog", "fish"]
    vectors = emb.embed(texts)
    assert len(vectors) == 3
    # Re-encoding individually should match the batch result (FastEmbed
    # is deterministic for the same input). This guards against any
    # accidental batch-order shuffle.
    for i, t in enumerate(texts):
        assert emb.embed_one(t) == vectors[i]


def test_embed_empty_string_does_not_crash():
    """The walker can emit Documents with whitespace-only content
    (very rare, but possible — empty `.gitkeep`-style files). The
    embedder must not raise."""
    emb = Embedder()
    vec = emb.embed_one("")
    assert len(vec) == DEFAULT_DIM * 4


def test_embed_empty_list_returns_empty_list():
    emb = Embedder()
    assert emb.embed([]) == []


def test_semantic_similarity_orders_paraphrases_above_unrelated():
    emb = Embedder()
    base = emb.embed_one("the user logs in with their password")
    paraphrase = emb.embed_one("authentication via credential input")
    unrelated = emb.embed_one("compile the rust binary with optimizations")
    sim_para = cosine_similarity(bytes_to_floats(base), bytes_to_floats(paraphrase))
    sim_unrel = cosine_similarity(bytes_to_floats(base), bytes_to_floats(unrelated))
    # Looking for clear separation, not absolute threshold — model
    # similarity values shift between revisions.
    assert sim_para > sim_unrel, (
        f"paraphrase ({sim_para:.4f}) should beat unrelated ({sim_unrel:.4f})"
    )
    assert sim_para - sim_unrel > 0.1


def test_get_default_returns_singleton():
    a = get_default()
    b = get_default()
    assert a is b


def test_reset_default_drops_singleton():
    a = get_default()
    reset_default()
    b = get_default()
    assert a is not b


def test_bytes_to_floats_round_trips():
    emb = Embedder()
    vec_bytes = emb.embed_one("round trip")
    floats = bytes_to_floats(vec_bytes)
    assert len(floats) == DEFAULT_DIM
    assert all(isinstance(f, float) for f in floats)


def test_bytes_to_floats_rejects_wrong_length():
    with pytest.raises(ValueError, match="buffer length"):
        bytes_to_floats(b"\x00" * 100, dim=DEFAULT_DIM)


def test_cosine_similarity_handles_zero_vectors():
    z = [0.0] * DEFAULT_DIM
    v = [1.0] + [0.0] * (DEFAULT_DIM - 1)
    # Both ways round — should never ZeroDivisionError on the empty-
    # string sentinel case.
    assert cosine_similarity(z, v) == 0.0
    assert cosine_similarity(v, z) == 0.0


def test_cosine_similarity_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_similarity_identical_vectors_is_one():
    v = [0.1, 0.2, 0.3]
    # Tiny float tolerance for the sqrt round-trip
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert cosine_similarity(v1, v2) == 0.0


def test_embedder_config_dataclass_is_frozen():
    """EmbedderConfig is frozen so callers can't mutate it after the
    embedder has loaded a model under it."""
    cfg = EmbedderConfig()
    with pytest.raises(AttributeError):
        cfg.model = "other/model"  # type: ignore[misc]


def test_embedder_unavailable_message_points_at_extra(monkeypatch):
    """If ``fastembed`` isn't importable, the error tells the user how
    to fix it. We simulate this by patching the import."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("simulated missing fastembed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reset_default()
    with pytest.raises(EmbedderUnavailableError, match="hermes-memory-mcp\\[embeddings\\]"):
        Embedder()
