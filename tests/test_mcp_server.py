"""End-to-end test of the MCP server's tool surface.

We don't spin up stdio (that requires a real client). Instead we build the
server in-process and invoke the registered list_tools + call_tool handlers
directly, which is exactly what the SDK does under the hood for stdio.

This is the a2 acceptance gate: tools are visible to MCP clients with their
schemas + descriptions, and every call returns a citation-shaped JSON payload.
"""

from __future__ import annotations

import json

import pytest
from mcp import types

from hermes_memory_mcp.schemas import TOOL_DESCRIPTIONS, TOOL_SCHEMAS
from hermes_memory_mcp.server import TOOLS, build_server


async def _list_tools(server) -> list[types.Tool]:
    handler = server.request_handlers[types.ListToolsRequest]
    req = types.ListToolsRequest(method="tools/list")
    resp = await handler(req)
    return resp.root.tools


async def _call_tool(server, name: str, arguments: dict) -> list[types.TextContent]:
    handler = server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=arguments),
    )
    resp = await handler(req)
    return resp.root.content


EXPECTED_TOOLS = {
    "search_memory",
    "get_project_brief",
    "find_decision",
    "what_changed_since",
    "check_claim_against_memory",
}


def test_all_five_tools_registered() -> None:
    assert set(TOOLS.keys()) == EXPECTED_TOOLS
    assert set(TOOL_SCHEMAS.keys()) == EXPECTED_TOOLS
    assert set(TOOL_DESCRIPTIONS.keys()) == EXPECTED_TOOLS


def test_every_schema_is_object_with_additional_properties_false() -> None:
    """Lock the schema shape so future tool additions don't accidentally
    drop the input validation rails."""
    for name, schema in TOOL_SCHEMAS.items():
        assert schema["type"] == "object", f"{name} schema is not an object"
        assert schema.get("additionalProperties") is False, (
            f"{name} schema must reject unknown keys (additionalProperties=False)"
        )
        assert schema.get("properties"), f"{name} schema must declare properties"


def test_descriptions_explain_when_to_use_the_tool() -> None:
    """Descriptions are what MCP clients show to agents during tool discovery.
    They must answer 'when should the agent call this?', not just 'what does
    it do?'. Heuristic: 'use this' should appear."""
    for name, desc in TOOL_DESCRIPTIONS.items():
        assert "use this" in desc.lower(), (
            f"{name} description must explain WHEN to use the tool, not just what it does"
        )


@pytest.mark.asyncio
async def test_list_tools_returns_all_five_with_schemas() -> None:
    server = build_server()
    tools = await _list_tools(server)
    names = {t.name for t in tools}
    assert names == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description, f"{tool.name} missing description"
        assert tool.inputSchema, f"{tool.name} missing inputSchema"
        # Sanity: schema matches what schemas.py declared
        assert tool.inputSchema == TOOL_SCHEMAS[tool.name]


@pytest.fixture
def isolated_index_root(tmp_path, monkeypatch):
    """Point search_memory at a tmp project root so tests never touch the
    user's real ~/.hermes-memory/ indexes. We set HERMES_MEMORY_ROOT to a
    fresh tmp dir, which the server's _resolve_project_root() reads."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HERMES_MEMORY_ROOT", str(project))
    # Also redirect default_index_dir so the index lives under tmp_path
    # rather than ~/.hermes-memory/.
    from hermes_memory_mcp import index as index_mod

    monkeypatch.setattr(index_mod, "default_index_dir", lambda root: tmp_path / "_index")
    return project


@pytest.mark.asyncio
async def test_search_memory_round_trip_returns_cited_payload(isolated_index_root) -> None:
    """With an empty index, search_memory returns a citation-shaped payload
    explaining the index is empty. Confirms the tool wires through the SDK
    correctly without polluting the user's real index."""
    server = build_server()
    content = await _call_tool(server, "search_memory", {"query": "test"})
    assert len(content) == 1
    payload = json.loads(content[0].text)
    assert "content" in payload
    assert "citations" in payload
    assert payload["citations"] == []
    # Empty-index path triggers "No results" via empty_result()
    assert "No results" in payload["content"]


@pytest.mark.asyncio
async def test_check_claim_round_trip_returns_cited_payload() -> None:
    server = build_server()
    content = await _call_tool(server, "check_claim_against_memory", {"claim": "the sky is green"})
    payload = json.loads(content[0].text)
    assert payload["citations"] == []
    assert "No results" in payload["content"]


@pytest.mark.asyncio
async def test_all_five_tools_invocable_through_sdk(isolated_index_root) -> None:
    """Every tool must round-trip through the SDK handlers and return a
    citation-shaped JSON payload. This is the a2 acceptance gate; a3 wired
    search_memory to a real index so we use isolated_index_root."""
    server = build_server()
    sample_args = {
        "search_memory": {"query": "smoke test"},
        "get_project_brief": {},
        "find_decision": {"topic": "smoke test"},
        "what_changed_since": {"reference": "last_session"},
        "check_claim_against_memory": {"claim": "smoke test"},
    }
    for name in EXPECTED_TOOLS:
        content = await _call_tool(server, name, sample_args[name])
        assert len(content) == 1
        payload = json.loads(content[0].text)
        assert "content" in payload, f"{name} payload missing 'content'"
        assert "citations" in payload, f"{name} payload missing 'citations'"
        assert isinstance(payload["citations"], list), f"{name} citations must be a list"
