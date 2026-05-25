"""Walker tests — make sure we classify, ignore, and read files correctly
without touching the real filesystem outside tmp_path fixtures."""

from __future__ import annotations

from pathlib import Path

from hermes_memory_mcp.walker import (
    DEFAULT_IGNORE_DIRS,
    DOC_TYPE_ADR,
    DOC_TYPE_CODE,
    DOC_TYPE_LOG,
    DOC_TYPE_MARKDOWN,
    DOC_TYPE_OTHER,
    Document,
    classify,
    walk,
)


def test_classify_markdown(tmp_path: Path) -> None:
    assert classify(tmp_path / "README.md") == DOC_TYPE_MARKDOWN
    assert classify(tmp_path / "notes.markdown") == DOC_TYPE_MARKDOWN
    assert classify(tmp_path / "doc.mdx") == DOC_TYPE_MARKDOWN


def test_classify_adr_by_parent_dir(tmp_path: Path) -> None:
    assert classify(tmp_path / "docs" / "adr" / "0001-something.md") == DOC_TYPE_ADR
    assert classify(tmp_path / "decisions" / "0002.md") == DOC_TYPE_ADR
    assert classify(tmp_path / "ADRS" / "ABC.md") == DOC_TYPE_ADR


def test_classify_code(tmp_path: Path) -> None:
    assert classify(tmp_path / "main.py") == DOC_TYPE_CODE
    assert classify(tmp_path / "main.rs") == DOC_TYPE_CODE
    assert classify(tmp_path / "app.tsx") == DOC_TYPE_CODE
    assert classify(tmp_path / "Cargo.toml") == DOC_TYPE_CODE


def test_classify_log(tmp_path: Path) -> None:
    assert classify(tmp_path / "build.log") == DOC_TYPE_LOG


def test_classify_other(tmp_path: Path) -> None:
    assert classify(tmp_path / "image.png") == DOC_TYPE_OTHER
    assert classify(tmp_path / "data.parquet") == DOC_TYPE_OTHER


def test_walk_yields_markdown_and_code(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Hello")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')")
    docs = list(walk(tmp_path))
    types = {d.doc_type for d in docs}
    assert types == {DOC_TYPE_MARKDOWN, DOC_TYPE_CODE}


def test_walk_skips_ignored_dirs(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Hello")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "stuff.js").write_text("module.exports={}")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    docs = list(walk(tmp_path))
    paths = [str(d.file_path) for d in docs]
    assert any("README.md" in p for p in paths)
    assert all("node_modules" not in p for p in paths)
    assert all(".git" not in p for p in paths)


def test_walk_respects_max_file_size(tmp_path: Path) -> None:
    (tmp_path / "big.md").write_text("x" * 2_000_000)
    (tmp_path / "small.md").write_text("# tiny")
    docs = list(walk(tmp_path, max_file_size=500_000))
    paths = [d.file_path.name for d in docs]
    assert "big.md" not in paths
    assert "small.md" in paths


def test_walk_extra_ignore(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# keep")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key.md").write_text("# secret")
    docs = list(walk(tmp_path, extra_ignore=frozenset({"secrets"})))
    paths = [d.file_path.name for d in docs]
    assert "keep.md" in paths
    assert "key.md" not in paths


def test_walk_classifies_adr_inside_docs_adr(tmp_path: Path) -> None:
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    (adr_dir / "0001-use-fts5.md").write_text("# ADR-0001\nUse FTS5 first")
    docs = list(walk(tmp_path))
    assert len(docs) == 1
    assert docs[0].doc_type == DOC_TYPE_ADR


def test_walk_other_type_skipped_by_default(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # binary header
    (tmp_path / "doc.md").write_text("# doc")
    docs = list(walk(tmp_path))
    assert {d.doc_type for d in docs} == {DOC_TYPE_MARKDOWN}

    # With include_other=True it would be visible — but PNG bytes are
    # readable as UTF-8 with errors="replace", so we'd get it back. Just
    # confirm the count goes up.
    docs_all = list(walk(tmp_path, include_other=True))
    assert len(docs_all) == 2


def test_document_dataclass_is_frozen() -> None:
    doc = Document(
        file_path=Path("/x"),
        doc_type=DOC_TYPE_MARKDOWN,
        content="hi",
        mtime=1.0,
        size=2,
    )
    import pytest

    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
        doc.content = "no"  # type: ignore[misc]


def test_default_ignore_dirs_includes_common_build_outputs() -> None:
    expected = {".git", "node_modules", "__pycache__", "target", "dist", "build"}
    assert expected <= DEFAULT_IGNORE_DIRS
