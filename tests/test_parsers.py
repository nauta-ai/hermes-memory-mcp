"""Parser tests — frontmatter, ADR status, links."""

from __future__ import annotations

from hermes_memory_mcp.parsers import (
    AdrStatus,
    extract_links,
    parse_adr_status,
    parse_frontmatter,
)

# ── frontmatter ───────────────────────────────────────────────────────────


def test_frontmatter_basic_kv() -> None:
    text = "---\ntitle: My Note\nstatus: active\n---\n# Body\nhello"
    meta, body = parse_frontmatter(text)
    assert meta == {"title": "My Note", "status": "active"}
    assert body.startswith("# Body")


def test_frontmatter_quoted_strings_preserve_internal_punct() -> None:
    text = '---\ntitle: "Hello: world"\n---\nbody'
    meta, _ = parse_frontmatter(text)
    assert meta["title"] == "Hello: world"


def test_frontmatter_inline_list() -> None:
    text = "---\ntags: [python, mcp, memory]\n---\nbody"
    meta, _ = parse_frontmatter(text)
    assert meta["tags"] == ["python", "mcp", "memory"]


def test_frontmatter_boolean_and_null() -> None:
    text = "---\npublished: true\ndraft: false\nreviewer: null\n---\nbody"
    meta, _ = parse_frontmatter(text)
    assert meta == {"published": True, "draft": False, "reviewer": None}


def test_frontmatter_numbers() -> None:
    text = "---\norder: 42\nweight: 3.14\n---\nbody"
    meta, _ = parse_frontmatter(text)
    assert meta == {"order": 42, "weight": 3.14}


def test_frontmatter_missing_returns_empty_dict() -> None:
    text = "# Just a heading\nNo frontmatter here."
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_frontmatter_handles_blank_and_comment_lines() -> None:
    text = "---\n# this is a comment\ntitle: x\n\nstatus: y\n---\nbody"
    meta, _ = parse_frontmatter(text)
    assert meta == {"title": "x", "status": "y"}


def test_frontmatter_body_unchanged_when_block_present() -> None:
    text = "---\nkey: val\n---\nfirst body line\nsecond"
    _, body = parse_frontmatter(text)
    assert body == "first body line\nsecond"


# ── ADR status ────────────────────────────────────────────────────────────


def test_adr_status_accepted() -> None:
    text = "# ADR-0001\n## Status\nAccepted\n\n## Decision\nWe will..."
    status = parse_adr_status(text)
    assert status is not None
    assert status.state == "accepted"
    assert status.superseded_by is None


def test_adr_status_superseded_by_id() -> None:
    text = "# ADR-0007\n## Status\nSuperseded by ADR-0042\n\n## Context\n..."
    status = parse_adr_status(text)
    assert status.state == "superseded"
    assert status.superseded_by == "0042"


def test_adr_status_deprecated() -> None:
    text = "## Status\nDeprecated as of 2026-04-15"
    status = parse_adr_status(text)
    assert status.state == "deprecated"


def test_adr_status_proposed() -> None:
    text = "## Status\nProposed"
    status = parse_adr_status(text)
    assert status.state == "proposed"


def test_adr_status_missing_returns_none() -> None:
    text = "# Just a doc\n\nNo status section here."
    assert parse_adr_status(text) is None


def test_adr_status_is_frozen() -> None:
    s = AdrStatus(state="accepted", superseded_by=None, raw="Accepted")
    import pytest

    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
        s.state = "deprecated"  # type: ignore[misc]


# ── links ─────────────────────────────────────────────────────────────────


def test_extract_wiki_links() -> None:
    text = "See [[other-note]] and [[also-this|with display]]."
    links = extract_links(text)
    assert sorted(links.wiki_links) == ["also-this", "other-note"]


def test_extract_markdown_links_filters_external() -> None:
    text = """
    Local: [a](./docs/a.md)
    External: [b](https://example.com)
    Anchor: [c](#section)
    Email: [d](mailto:x@y.z)
    With fragment: [e](docs/b.md#anchor)
    """
    links = extract_links(text)
    md = links.markdown_links
    assert "./docs/a.md" in md
    assert "docs/b.md" in md  # fragment stripped
    assert "https://example.com" not in md
    assert all(not p.startswith("#") for p in md)
    assert all(not p.startswith("mailto:") for p in md)


def test_extract_links_iterates_all() -> None:
    text = "[[wiki]] and [md](./local.md)"
    links = extract_links(text)
    assert list(links) == ["wiki", "./local.md"]
