"""Hermes Memory MCP — MCP server entry point.

Exposes 5 tools per the v1 spec:
    search_memory(query, scope="all", limit=10)
    get_project_brief(repo_or_topic="current", as_of="now")
    find_decision(topic)
    what_changed_since(reference)
    check_claim_against_memory(claim)

Every tool response routes through ``cite.CitedResponse`` so the citation
contract is enforced at the API boundary.

v0.1.0a1: tool stubs return placeholder responses. The real implementation
wires each tool to the corresponding backend (Memory V3 KG search, the
Karpathy wiki compiler output, Cassandra-style contradiction check, etc.)
in subsequent sessions.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from .cite import Citation, CitedResponse, empty_result


async def search_memory(query: str, scope: str = "all", limit: int = 10) -> dict:
    """Free-text semantic search across the user's memory corpus.

    Scope: 'all' | 'notes' | 'decisions' | 'logs' | 'code' | 'git'.
    Returns up to ``limit`` ranked passages with citations.
    """
    # v0.1.0a1 stub — real impl: load local SQLite vector index, run query,
    # return ranked passages. For now, return an empty result so MCP clients
    # can wire the tool and see the contract shape.
    if not query.strip():
        return empty_result("empty query").to_dict()
    return empty_result(f"v0.1.0a1 stub: search not yet implemented (query={query!r}, scope={scope}, limit={limit})").to_dict()


async def get_project_brief(repo_or_topic: str = "current", as_of: str = "now") -> dict:
    """Return a current-state wiki brief on a topic, compiled (not generated)."""
    return empty_result(f"v0.1.0a1 stub: brief compilation not yet implemented (topic={repo_or_topic}, as_of={as_of})").to_dict()


async def find_decision(topic: str) -> dict:
    """Return the full decision chain on a topic, including reversals."""
    return empty_result(f"v0.1.0a1 stub: decision search not yet implemented (topic={topic!r})").to_dict()


async def what_changed_since(reference: str) -> dict:
    """Diff the current memory snapshot against a prior reference.

    ``reference`` accepts 'last_session', an ISO date 'YYYY-MM-DD', or a
    named snapshot id. Returns new/modified entries grouped by doc_type.
    """
    return empty_result(f"v0.1.0a1 stub: diff snapshot not yet implemented (reference={reference!r})").to_dict()


async def check_claim_against_memory(claim: str) -> dict:
    """Cassandra-style contradiction check.

    Given a claim the agent is about to make or act on, surface
    contradicting sources, code evidence, and a verdict.
    """
    return empty_result(f"v0.1.0a1 stub: claim verification not yet implemented (claim={claim!r})").to_dict()


# ── MCP stdio server wiring ────────────────────────────────────────────────

# The actual MCP protocol wiring (stdio transport, JSON-RPC tool registration,
# typed schemas) lives here. In v0.1.0a1 we keep the implementation minimal:
# tools are functions; a thin dispatcher routes JSON-RPC calls to them. The
# next session integrates the official ``mcp`` Python SDK for the spec-correct
# transport + schema generation.


TOOLS: dict[str, Any] = {
    "search_memory": search_memory,
    "get_project_brief": get_project_brief,
    "find_decision": find_decision,
    "what_changed_since": what_changed_since,
    "check_claim_against_memory": check_claim_against_memory,
}


async def _async_main() -> int:
    """Entry point for ``hermes-memory-mcp`` console script.

    v0.1.0a1: prints a placeholder banner + the list of registered tools, then
    exits. v0.1.0a2 will replace this with the real MCP stdio loop.
    """
    print("hermes-memory-mcp v0.1.0a1 — local memory MCP for AI coding agents", file=sys.stderr)
    print("(alpha — tools registered but stubs only; not a functional MCP server yet)", file=sys.stderr)
    print("Registered tools:", file=sys.stderr)
    for name in TOOLS:
        print(f"  - {name}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Next session: wire the official `mcp` SDK stdio loop + per-tool schemas.", file=sys.stderr)
    return 0


def main() -> int:
    """Sync entry point — console_script target."""
    return asyncio.run(_async_main())


if __name__ == "__main__":
    sys.exit(main())
