# Hermes Memory MCP

Local-first MCP server that gives AI coding agents persistent, cited memory of your project's notes, decisions, commits, and logs.

**Status:** v0.1.0a1 — alpha scaffold. Tool surfaces are registered; full implementations land in subsequent releases. Tracks the benchmark at [Claude Code Amnesia Test](https://github.com/nauta-ai/claude-code-amnesia-test).

## What it does

Five MCP tools, brutally simple:

```
search_memory(query, scope="all", limit=10)        # ranked passages from your corpus
get_project_brief(repo_or_topic, as_of="now")      # current state of a topic, cited
find_decision(topic)                                # the actual ADR + any reversal chain
what_changed_since(reference)                       # diff against the last compile
check_claim_against_memory(claim)                   # flag stale or contradicted assumptions
```

Every response includes `file_path` + `line_range` citations. No prose answers without sources.

## Status milestones

- **v0.1.0a1 (this release)** — package scaffold, tool surfaces, citation contract, 6 tests passing
- **v0.1.0a2** — official MCP SDK stdio loop wired; per-tool JSON schemas; first end-to-end with Claude Desktop
- **v0.1.0a3** — filesystem walker + Markdown + ADR parsers; local SQLite vector index
- **v0.1.0a4** — `install-mcp` writers for Claude Desktop / Cursor / Cline configs
- **v0.1.0a5** — end-to-end demo against the [Claude Code Amnesia Test](https://github.com/nauta-ai/claude-code-amnesia-test) sample project, ≥22/25 score
- **v0.1.0** — stable, install-ready

## Install (preview only — not yet useful)

```bash
pip install hermes-memory-mcp
hermes-memory version
```

## License

MIT
