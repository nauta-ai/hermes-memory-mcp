"""Hermes Memory MCP — local, cited memory for AI coding agents.

Public API:
    Memory      — the indexed corpus + query surface
    cite        — citation contract enforcement
    run_server  — start the MCP stdio server

CLI entry points (see pyproject.toml [project.scripts]):
    hermes-memory      — interactive CLI (init, ask, install-mcp)
    hermes-memory-mcp  — MCP server stdin/stdout entry
"""

__version__ = "0.1.0a5"
__all__ = ["__version__"]
