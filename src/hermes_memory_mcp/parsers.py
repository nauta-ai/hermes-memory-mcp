"""Document parsers — extract structured metadata from raw file content.

a4 adds:
* ``parse_frontmatter`` — YAML frontmatter from Markdown files (Obsidian
  vault notes, MkDocs, Hugo, etc.). No PyYAML dependency: we parse the
  subset that matters (key: value, key: [list, of, things], booleans,
  dates as strings). Robust enough for real notes; deliberately not a
  general YAML parser.

* ``parse_adr_status`` — extract Status field from a Madr/Nygard ADR.
  Returns ('accepted', None) for plain status, or ('superseded', 'ADR-0042')
  for reversal-chain entries. Powers find_decision's reversal awareness.

* ``extract_links`` — pull out [[wiki-link]] and (markdown)[#anchor] refs.
  Used by the future Cassandra graph layer to build cross-doc edges.

Every parser is a pure function: ``content: str -> structured``. Walker
calls them on the way to the index, so parsing happens once at ingest
not at query time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── frontmatter ───────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Strip leading YAML frontmatter and return (metadata, remaining_body).

    No metadata block? Returns ({}, content) unchanged. Malformed YAML?
    Returns whatever could be parsed plus the body after the closing ---.

    Supported value types:
      - scalars: strings (quoted or bare), booleans (true/false),
        integers, floats
      - lists: ``[a, b, c]`` inline only (block form rare in notes)
      - nested dicts: not supported in a4 — frontmatter in coding-note
        corpora is overwhelmingly flat. Skipped values keep their raw form.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    block = match.group(1)
    body = content[match.end() :]
    metadata: dict[str, Any] = {}

    for line in block.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        metadata[key] = _coerce_scalar(value)

    return metadata, body


def _coerce_scalar(raw: str) -> Any:
    """Best-effort scalar coercion. Returns the raw string if nothing
    else fits."""
    if not raw:
        return ""

    # Quoted string — strip quotes
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]

    # Inline list: [a, b, c]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(item.strip()) for item in inner.split(",")]

    low = raw.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None

    # Number
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    return raw


# ── ADR status ────────────────────────────────────────────────────────────

# Common ADR status section headers (Nygard, Madr, etc.)
_STATUS_RE = re.compile(
    r"^\s*##?\s*Status\s*\n+(.*?)(?:\n\s*##|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# 'Superseded by ADR-0042' / 'Supersedes #41' / 'Reverted by 0045'
_SUPERSEDED_BY_RE = re.compile(
    r"\bsupersed(?:ed|es|ing)?\s+(?:by\s+)?[#-]?(?:ADR[-_])?(\d+|[A-Za-z][\w-]+)",
    re.IGNORECASE,
)
_REVERTED_BY_RE = re.compile(
    r"\b(?:reverted|withdrawn|rejected)\s+(?:by\s+)?[#-]?(?:ADR[-_])?(\d+|[A-Za-z][\w-]+)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AdrStatus:
    """Outcome of parsing an ADR's Status section.

    state         — one of 'accepted', 'proposed', 'deprecated', 'superseded',
                    'rejected', 'reverted', 'unknown'
    superseded_by — ID of the ADR that replaces this one, if any (e.g. '0042')
    raw           — verbatim text of the Status section (for citation)
    """

    state: str
    superseded_by: str | None
    raw: str


def parse_adr_status(content: str) -> AdrStatus | None:
    """Parse the Status section of an ADR. Returns None if no Status
    section is found."""
    match = _STATUS_RE.search(content)
    if not match:
        return None

    raw = match.group(1).strip()
    low = raw.lower()

    superseded_by = None
    sm = _SUPERSEDED_BY_RE.search(low)
    if sm:
        superseded_by = sm.group(1)

    if "supersed" in low and "by" in low.split("supersed", 1)[1][:20]:
        state = "superseded"
    elif "deprecat" in low:
        state = "deprecated"
    elif _REVERTED_BY_RE.search(low):
        state = "reverted"
    elif "rejected" in low:
        state = "rejected"
    elif "propos" in low:
        state = "proposed"
    elif "accept" in low or "approved" in low:
        state = "accepted"
    else:
        state = "unknown"

    return AdrStatus(state=state, superseded_by=superseded_by, raw=raw)


# ── link extraction ───────────────────────────────────────────────────────

_WIKI_LINK_RE = re.compile(r"\[\[([^\[\]|]+?)(?:\|[^\[\]]+)?\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[(?:[^\]]+)\]\(([^)]+)\)")


@dataclass
class DocLinks:
    """All outbound link targets discovered in a document."""

    wiki_links: list[str] = field(default_factory=list)
    markdown_links: list[str] = field(default_factory=list)

    def __iter__(self):
        yield from self.wiki_links
        yield from self.markdown_links


def extract_links(content: str) -> DocLinks:
    """Return all link targets in ``content``. Filters out anchor-only
    links (``#section``) and absolute URLs (``http://...``); only
    intra-corpus references make it into the result."""
    wiki = _WIKI_LINK_RE.findall(content)

    md = []
    for raw in _MARKDOWN_LINK_RE.findall(content):
        target = raw.strip()
        # Drop external URLs + pure anchors — the graph layer wants
        # local doc references only.
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        # Drop the fragment if present
        if "#" in target:
            target = target.split("#", 1)[0]
        if target:
            md.append(target)

    return DocLinks(wiki_links=wiki, markdown_links=md)
