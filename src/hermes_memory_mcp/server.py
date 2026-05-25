"""Hermes Memory MCP — MCP server entry point.

v0.1.0a2: official MCP SDK stdio loop wired. Each tool surface in the v1 spec
is registered with a per-tool JSON Schema (see ``schemas.py``). Tool bodies
still return stub responses — the index/walker/parser layer lands in a3.

The server speaks the standard MCP stdio transport, so it works out of the
box with Claude Desktop, Cursor, Cline, and any other client that ships with
the spec-compliant stdio adapter. Wire it via the operator CLI:

    hermes-memory install-mcp claude-desktop

Citation contract: every tool response goes through ``cite.CitedResponse``,
serialized as a single TextContent block carrying JSON. MCP clients can
render the JSON, and downstream tooling can parse the citations field
directly without scraping prose.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from .cite import Citation, CitedResponse, empty_result
from .index import Index
from .schemas import TOOL_DESCRIPTIONS, TOOL_SCHEMAS

SERVER_NAME = "hermes-memory-mcp"
SERVER_VERSION = "0.1.0a4"


def _resolve_project_root() -> Path:
    """Project root whose index this server queries.

    Precedence: ``HERMES_MEMORY_ROOT`` env var → current working directory.
    The MCP client (Claude Desktop / Cursor / Cline) sets cwd when it
    spawns us, so cwd usually matches the user's active project. Env var
    is the explicit override for multi-project setups."""
    env = os.environ.get("HERMES_MEMORY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


# ── Tool implementations (a2: stubs returning CitedResponse-shaped JSON) ───
#
# These are deliberately small. The MCP SDK validates inputs against the
# JSON Schemas registered below before invoking, so the tool bodies can
# assume well-formed arguments.


async def _search_memory(query: str, scope: str = "all", limit: int = 10) -> CitedResponse:
    if not query.strip():
        return empty_result("empty query")
    root = _resolve_project_root()
    try:
        with Index.open(root) as ix:
            if ix.doc_count() == 0:
                return empty_result(
                    f"index at {ix.db_path} is empty — run `hermes-memory init {root}` first"
                )
            hits = ix.search(query, scope=scope, limit=limit)
    except RuntimeError as exc:
        # e.g. schema mismatch on stale index
        return empty_result(f"index error: {exc}")
    if not hits:
        return empty_result(f"no hits for {query!r} (scope={scope})")

    lines = [f"Found {len(hits)} hits for {query!r} (scope={scope}):"]
    citations: list[Citation] = []
    for i, hit in enumerate(hits, 1):
        # Snippet from FTS5 includes <<<...>>> markers around matched
        # tokens; preserve them so agents can see where the hit landed.
        snippet = " ".join(hit.snippet.split())
        lines.append(f"[{i}] {hit.file_path} ({hit.doc_type})")
        lines.append(f"    {snippet}")
        citations.append(Citation(file_path=hit.file_path, line_range=None, snippet=snippet))
    return CitedResponse(content="\n".join(lines), citations=citations)


async def _get_project_brief(repo_or_topic: str = "current", as_of: str = "now") -> CitedResponse:
    """Compile a deterministic brief about the project.

    a3 brief composition:
      1. README file (root-level *.md picked by filename heuristic)
      2. Doc-type breakdown (how many ADRs, code files, etc.)
      3. Most-recently-modified docs across all types (top 10)
      4. If ``repo_or_topic`` != 'current', also folds in the top ADR + top
         code hits matching that string.

    No LLM in the loop — the brief is assembled from indexed metadata + a
    single excerpt from the README. Output stays the same across runs
    given the same corpus, which is the contract that distinguishes
    'brief' from 'summary'."""
    root = _resolve_project_root()
    try:
        with Index.open(root) as ix:
            if ix.doc_count() == 0:
                return empty_result(
                    f"index at {ix.db_path} is empty — run `hermes-memory init {root}` first"
                )

            # 1. README (root-level markdown that looks like a top-level doc)
            readme_row = ix.conn.execute(
                """SELECT file_path FROM documents
                   WHERE doc_type = 'markdown'
                     AND (lower(file_path) LIKE '%/readme.md'
                          OR lower(file_path) LIKE '%/readme.markdown'
                          OR lower(file_path) LIKE '%/readme.mdx')
                   ORDER BY length(file_path) ASC
                   LIMIT 1""",
            ).fetchone()

            # 2. Doc-type breakdown
            breakdown = ix.doc_types()

            # 3. Most-recently-modified files
            recent_rows = ix.conn.execute(
                """SELECT file_path, doc_type, mtime
                   FROM documents
                   ORDER BY mtime DESC
                   LIMIT 10""",
            ).fetchall()

            # 4. Topic-scoped hits if caller asked for one
            topic_hits = []
            if repo_or_topic and repo_or_topic.lower() != "current":
                topic_hits = ix.search(repo_or_topic, scope="all", limit=5)
    except RuntimeError as exc:
        return empty_result(f"index error: {exc}")

    lines: list[str] = []
    citations: list[Citation] = []

    lines.append(f"# Project brief — {root.name}")
    lines.append(f"_indexed at {ix.db_path} • {sum(breakdown.values())} documents_")
    lines.append("")

    # Breakdown
    lines.append("## Corpus")
    for doc_type, count in sorted(breakdown.items()):
        lines.append(f"- **{doc_type}**: {count}")
    lines.append("")

    # README excerpt
    if readme_row is not None:
        readme_path = readme_row["file_path"]
        try:
            text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        # Take the first ~40 non-blank lines as the excerpt; agents can
        # read the file itself for the full thing via the citation.
        excerpt_lines = [ln for ln in text.splitlines() if ln.strip()][:40]
        excerpt = "\n".join(excerpt_lines)
        lines.append("## README excerpt")
        lines.append(excerpt)
        lines.append("")
        citations.append(Citation(file_path=readme_path, line_range="1-40", snippet=None))

    # Recent files
    if recent_rows:
        from datetime import datetime

        lines.append("## Most recent changes")
        for row in recent_rows:
            mtime_str = datetime.fromtimestamp(row["mtime"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {mtime_str} — `{row['file_path']}` ({row['doc_type']})")
            citations.append(Citation(file_path=row["file_path"], line_range=None, snippet=None))
        lines.append("")

    # Topic hits
    if topic_hits:
        lines.append(f"## Hits matching '{repo_or_topic}'")
        for hit in topic_hits:
            snippet = " ".join(hit.snippet.split())
            lines.append(f"- `{hit.file_path}` ({hit.doc_type})")
            lines.append(f"  {snippet}")
            citations.append(Citation(file_path=hit.file_path, line_range=None, snippet=snippet))

    lines.append(f"_as_of: {as_of} (a3 ignores time anchoring; lands in a4)_")

    return CitedResponse(content="\n".join(lines), citations=citations)


async def _find_decision(topic: str) -> CitedResponse:
    """Return the decision chain for a topic.

    a4 layer: each matched ADR is opened on disk so we can parse its
    Status section (parsers.parse_adr_status). Output then groups hits
    into current (accepted/proposed/unknown) vs superseded/deprecated/
    reverted, so the agent sees which decision is in force right now."""
    if not topic.strip():
        return empty_result("empty topic")
    root = _resolve_project_root()
    try:
        with Index.open(root) as ix:
            if ix.doc_count() == 0:
                return empty_result(
                    f"index at {ix.db_path} is empty — run `hermes-memory init {root}` first"
                )
            # Prefix match each topic token so 'auth' matches 'authentication'.
            # FTS5 prefix syntax: token*
            import re as _re

            tokens = [t for t in _re.findall(r"[A-Za-z][A-Za-z0-9_-]+", topic) if t]
            if not tokens:
                return empty_result(f"no searchable tokens in {topic!r}")
            fts_query = " OR ".join(f"{t}*" for t in tokens)
            hits = ix.search(fts_query, scope="adr", limit=20, raw_fts=True)
    except RuntimeError as exc:
        return empty_result(f"index error: {exc}")
    if not hits:
        return empty_result(f"no ADRs match {topic!r}")

    from .parsers import parse_adr_status

    current: list[tuple] = []
    historical: list[tuple] = []
    for hit in hits:
        try:
            text = Path(hit.file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        status = parse_adr_status(text)
        bucket = (
            historical
            if (status and status.state in ("superseded", "deprecated", "reverted", "rejected"))
            else current
        )
        bucket.append((hit, status))

    lines = [f"# Decision chain for {topic!r}"]
    citations: list[Citation] = []

    def _render(label: str, group: list) -> None:
        if not group:
            return
        lines.append(f"\n## {label} ({len(group)})")
        for hit, status in group:
            snippet = " ".join(hit.snippet.split())
            state_str = status.state if status else "unknown"
            extra = (
                f" → superseded by {status.superseded_by}"
                if status and status.superseded_by
                else ""
            )
            lines.append(f"- **[{state_str}{extra}]** `{hit.file_path}`")
            lines.append(f"  {snippet}")
            citations.append(Citation(file_path=hit.file_path, line_range=None, snippet=snippet))

    _render("Currently in force", current)
    _render("Superseded / historical", historical)

    return CitedResponse(content="\n".join(lines), citations=citations)


def _parse_reference(reference: str) -> float | None:
    """Parse a 'what_changed_since' reference into a Unix timestamp.

    Accepts:
      - 'last_session' → 24h ago (heuristic)
      - 'YYYY-MM-DD'   → midnight that date, local time
      - 'NhNm' style relative → e.g. '2h', '30m', '7d'

    Returns None if the reference can't be parsed.
    """
    import re
    from datetime import datetime, timedelta

    ref = reference.strip().lower()
    if ref in ("last_session", "yesterday", "1d"):
        return (datetime.now() - timedelta(hours=24)).timestamp()

    # ISO date
    try:
        dt = datetime.strptime(ref, "%Y-%m-%d")
        return dt.timestamp()
    except ValueError:
        pass

    # Relative: 7d, 2h, 30m, 90s
    m = re.fullmatch(r"(\d+)\s*([smhd])", ref)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return (datetime.now() - timedelta(seconds=n * seconds)).timestamp()

    return None


async def _what_changed_since(reference: str) -> CitedResponse:
    """Return documents whose mtime is newer than ``reference``.

    Useful for catching up after time away: 'what changed since yesterday'
    returns the diff a developer needs to skim before resuming work."""
    cutoff = _parse_reference(reference)
    if cutoff is None:
        return empty_result(
            f"could not parse reference {reference!r}; try 'last_session', "
            "'YYYY-MM-DD', or relative '7d' / '2h' / '30m'"
        )

    root = _resolve_project_root()
    try:
        with Index.open(root) as ix:
            if ix.doc_count() == 0:
                return empty_result(
                    f"index at {ix.db_path} is empty — run `hermes-memory init {root}` first"
                )
            rows = ix.conn.execute(
                """SELECT file_path, doc_type, mtime
                   FROM documents
                   WHERE mtime > ?
                   ORDER BY mtime DESC
                   LIMIT 100""",
                (cutoff,),
            ).fetchall()
    except RuntimeError as exc:
        return empty_result(f"index error: {exc}")

    if not rows:
        return empty_result(f"no documents changed since {reference}")

    # Group by doc_type so the report is scannable
    by_type: dict[str, list] = {}
    for row in rows:
        by_type.setdefault(row["doc_type"], []).append(row)

    lines = [f"Found {len(rows)} changes since {reference}:"]
    citations: list[Citation] = []
    for doc_type, entries in sorted(by_type.items()):
        lines.append(f"\n## {doc_type} ({len(entries)})")
        for entry in entries:
            lines.append(f"  - {entry['file_path']}")
            citations.append(Citation(file_path=entry["file_path"], line_range=None, snippet=None))
    return CitedResponse(content="\n".join(lines), citations=citations)


async def _check_claim_against_memory(claim: str) -> CitedResponse:
    """Cassandra-style contradiction check.

    a3 implementation is deliberately conservative: we tokenize the claim
    into content words (drop stop-words + punctuation), search the
    corpus for passages containing those words, and surface any passage
    where a negation cue ('no', 'not', 'never', 'OBSOLETE', 'deprecated',
    'wrong', 'removed', 'reverted', 'parked') appears within ~10 tokens
    of a content word. We never claim 'X is false' — we just hand the
    agent the strongest candidate sources and let it judge.

    a4 layers semantic embedding + ADR-aware reversal-chain checks on
    top. The contract stays the same: a verdict word + citations."""
    if not claim.strip():
        return empty_result("empty claim")
    root = _resolve_project_root()
    try:
        with Index.open(root) as ix:
            if ix.doc_count() == 0:
                return empty_result(
                    f"index at {ix.db_path} is empty — run `hermes-memory init {root}` first"
                )
            # Pull the content words from the claim. Heuristic stop-word
            # list; aggressive enough to cut the noise without needing
            # nltk. Sort by length descending so longest (most specific)
            # word leads the query.
            import re

            stopwords = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "to",
                "of",
                "in",
                "on",
                "at",
                "for",
                "with",
                "and",
                "or",
                "but",
                "not",
                "no",
                "this",
                "that",
                "these",
                "those",
                "it",
                "its",
                "as",
                "by",
                "from",
                "we",
                "i",
                "you",
                "they",
                "he",
                "she",
                "do",
                "does",
                "did",
                "has",
                "have",
                "had",
                "will",
                "would",
                "should",
                "could",
                "may",
                "might",
                "can",
                "so",
                "if",
                "than",
                "then",
                "there",
                "here",
                "what",
                "which",
            }
            tokens = [
                t.lower()
                for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", claim)
                if t.lower() not in stopwords and len(t) > 2
            ]
            tokens.sort(key=len, reverse=True)
            tokens = tokens[:5]  # cap query breadth

            if not tokens:
                return empty_result("claim has no content words after stop-word filter")

            query = " OR ".join(tokens)
            hits = ix.search(query, scope="all", limit=20, raw_fts=True)
    except RuntimeError as exc:
        return empty_result(f"index error: {exc}")

    if not hits:
        return CitedResponse(
            content=(
                f"## verdict: UNKNOWN\n\n"
                f"No passages in the indexed corpus mention "
                f"any of: {', '.join(tokens)}.\n\n"
                f"The agent should not treat the claim as supported by memory."
            ),
            citations=[Citation(file_path=str(root), line_range=None, snippet=None)],
        )

    # Negation cue detection in snippets
    negation_cues = (
        "obsolete",
        "deprecated",
        "retired",
        "removed",
        "reverted",
        "parked",
        "never",
        " not ",
        "no longer",
        "decommissioned",
        "wrong",
        "incorrect",
        "rejected",
        "abandoned",
        "supersed",
    )
    flagged = []
    supporting = []
    for hit in hits:
        low = hit.snippet.lower()
        if any(cue in low for cue in negation_cues):
            flagged.append(hit)
        else:
            supporting.append(hit)

    verdict = "CONTRADICTED" if flagged else ("SUPPORTED" if supporting else "UNKNOWN")

    lines = [f"## verdict: {verdict}", ""]
    citations: list[Citation] = []

    if flagged:
        lines.append(f"### Contradicting / cautionary sources ({len(flagged)})")
        for hit in flagged[:8]:
            snippet = " ".join(hit.snippet.split())
            lines.append(f"- `{hit.file_path}` ({hit.doc_type})")
            lines.append(f"  {snippet}")
            citations.append(Citation(file_path=hit.file_path, line_range=None, snippet=snippet))
        lines.append("")

    if supporting:
        lines.append(f"### Other matching sources ({len(supporting)})")
        for hit in supporting[:5]:
            snippet = " ".join(hit.snippet.split())
            lines.append(f"- `{hit.file_path}` ({hit.doc_type})")
            lines.append(f"  {snippet}")
            citations.append(Citation(file_path=hit.file_path, line_range=None, snippet=snippet))
        lines.append("")

    lines.append(
        "_a3 verdict is heuristic: presence of negation cues near matched terms. "
        "Always have the agent skim the cited passages before acting on the verdict._"
    )

    return CitedResponse(content="\n".join(lines), citations=citations)


TOOLS: dict[str, callable] = {
    "search_memory": _search_memory,
    "get_project_brief": _get_project_brief,
    "find_decision": _find_decision,
    "what_changed_since": _what_changed_since,
    "check_claim_against_memory": _check_claim_against_memory,
}


# ── MCP server wiring ──────────────────────────────────────────────────────


def build_server() -> Server:
    """Return a configured MCP Server with all 5 tools registered.

    Factored out of ``run_stdio`` so tests can drive the server without
    binding to stdio. The next milestone (a3) will use this to run a
    smoke test that round-trips a tool call through the SDK in-process.
    """
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=TOOL_DESCRIPTIONS[name],
                inputSchema=TOOL_SCHEMAS[name],
            )
            for name in TOOLS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        impl = TOOLS.get(name)
        if impl is None:
            # SDK normally guards against this, but be explicit so the
            # error surfaces with a citation-shaped response anyway.
            payload = empty_result(f"unknown tool: {name!r}").to_dict()
            return [types.TextContent(type="text", text=json.dumps(payload))]

        result: CitedResponse = await impl(**(arguments or {}))
        return [types.TextContent(type="text", text=json.dumps(result.to_dict()))]

    return server


async def run_stdio() -> int:
    """Run the MCP server over stdio. Blocks until the client disconnects.

    This is what Claude Desktop / Cursor / Cline launch when the user wires
    hermes-memory-mcp into their MCP config. They spawn the binary and
    talk JSON-RPC over the process's stdin/stdout.
    """
    server = build_server()
    init_opts = InitializationOptions(
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, init_opts)
    return 0


def main() -> int:
    """Sync entry point — installed as the ``hermes-memory-mcp`` console script."""
    try:
        return asyncio.run(run_stdio())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
