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
SERVER_VERSION = "0.1.0a3"


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
    return empty_result(
        f"v0.1.0a2 stub: brief compiler not yet wired (topic={repo_or_topic}, as_of={as_of})"
    )


async def _find_decision(topic: str) -> CitedResponse:
    """Search the ADR corpus only. Returns every ADR mentioning the topic
    so the agent can see the original decision + any reversals.

    a4 will add explicit reversal-chain detection (parse 'Supersedes #N'
    style links). For now we return all hits and let the agent reason."""
    if not topic.strip():
        return empty_result("empty topic")
    root = _resolve_project_root()
    try:
        with Index.open(root) as ix:
            if ix.doc_count() == 0:
                return empty_result(
                    f"index at {ix.db_path} is empty — run `hermes-memory init {root}` first"
                )
            hits = ix.search(topic, scope="adr", limit=20)
    except RuntimeError as exc:
        return empty_result(f"index error: {exc}")
    if not hits:
        return empty_result(f"no ADRs match {topic!r}")

    lines = [f"Found {len(hits)} ADRs matching {topic!r}:"]
    citations: list[Citation] = []
    for i, hit in enumerate(hits, 1):
        snippet = " ".join(hit.snippet.split())
        lines.append(f"[{i}] {hit.file_path}")
        lines.append(f"    {snippet}")
        citations.append(Citation(file_path=hit.file_path, line_range=None, snippet=snippet))
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
    return empty_result(f"v0.1.0a2 stub: contradiction check not yet wired (claim={claim!r})")


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
