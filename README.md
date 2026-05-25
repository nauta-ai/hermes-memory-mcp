# Hermes Memory MCP

Local-first MCP server that gives AI coding agents persistent, cited memory of your project's notes, decisions, commits, and logs.

**Status:** v0.1.0a3 — real filesystem walker, SQLite FTS5 index, **three of five tools wired end-to-end** (`search_memory`, `find_decision`, `what_changed_since`). The other two land in a4. Tracks the benchmark at [Claude Code Amnesia Test](https://github.com/nauta-ai/claude-code-amnesia-test).

## What it does

Five MCP tools, brutally simple:

```
search_memory(query, scope="all", limit=10)        # ranked passages from your corpus
get_project_brief(repo_or_topic, as_of="now")      # current state of a topic, cited
find_decision(topic)                                # the actual ADR + any reversal chain
what_changed_since(reference)                       # diff against the last compile
check_claim_against_memory(claim)                   # flag stale or contradicted assumptions
```

Every response includes `file_path` (+ `line_range` once a4 lands) citations. No prose answers without sources.

## Quickstart

```bash
# Install from this repo (PyPI publish lands in a5)
pip install -e .

# Index a project — walks the filesystem, classifies files (markdown / adr /
# code / log), and writes a SQLite FTS5 index to ~/.hermes-memory/<hash>/.
hermes-memory init ~/projects/my-app

# Search it directly, no MCP client needed
hermes-memory ask "authentication design" --root ~/projects/my-app

# Wire the MCP server into your client. Reads existing config, splices in
# the hermes-memory-mcp entry, leaves every other server entry untouched.
hermes-memory install-mcp claude-desktop
# (or: cursor, cline)
```

After `install-mcp`, restart your client. The five tools appear in the tool list automatically.

## Why local-first

Every existing agent-memory MCP server I evaluated either (a) ships data to a third-party API, (b) requires a cloud index, or (c) requires extensive infra setup. None of those work for projects under NDA, on air-gapped networks, or for solo devs who don't want their notes leaving their laptop.

This server runs entirely on the user's machine. The index is a plain SQLite file. The corpus is the user's filesystem. The MCP transport is stdio. No network calls happen unless the user explicitly adds an embedding model that calls out (and a4 ships with a local default).

## Status milestones

- **v0.1.0a1** — package scaffold, tool surfaces, citation contract, 6 tests passing
- **v0.1.0a2** — official MCP SDK stdio loop, per-tool JSON schemas, install-mcp writers for Claude Desktop / Cursor / Cline, 21 tests passing
- **v0.1.0a3 (this release)** — filesystem walker, SQLite FTS5 index, working `search_memory` + `find_decision` + `what_changed_since` tools + `hermes-memory init`/`ask` CLI, **49 tests passing**
- **v0.1.0a4** — Markdown frontmatter + ADR parsing; `get_project_brief` + `check_claim_against_memory` wired; optional local embeddings on top of FTS5 for semantic recall
- **v0.1.0a5** — end-to-end demo against the [Claude Code Amnesia Test](https://github.com/nauta-ai/claude-code-amnesia-test) sample project, ≥22/25 score; PyPI publish
- **v0.1.0** — stable, install-ready

## License

MIT
