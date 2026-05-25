# Hermes Memory MCP

Local-first MCP server that gives AI coding agents persistent, cited memory of your project's notes, decisions, commits, and logs.

**Status:** v0.1.0a5 — adds **local semantic embeddings** (FastEmbed / ONNX, ~80 MB, no PyTorch) blended with FTS5 BM25 via Reciprocal Rank Fusion for hybrid lexical + semantic recall. Opt-in via the `[embeddings]` extra; the FTS-only path keeps working unchanged. 101 tests passing. Tracks the benchmark at [Claude Code Amnesia Test](https://github.com/nauta-ai/claude-code-amnesia-test).

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
# Install
pip install hermes-memory-mcp

# Optional: add the semantic embeddings extra (FastEmbed / ONNX, ~80 MB)
pip install 'hermes-memory-mcp[embeddings]'

# Index a project — walks the filesystem, classifies files (markdown / adr /
# code / log), and writes a SQLite FTS5 index to ~/.hermes-memory/<hash>/.
hermes-memory init ~/projects/my-app

# (a5) Populate embeddings on top of the index for semantic recall.
# Safe to re-run — only the NULL-embedding docs are encoded.
hermes-memory embed --root ~/projects/my-app

# Search it directly, no MCP client needed. --embeddings turns on hybrid
# FTS5 + cosine ranking via Reciprocal Rank Fusion.
hermes-memory ask "how do users sign in" --root ~/projects/my-app --embeddings

# Wire the MCP server into your client. Reads existing config, splices in
# the hermes-memory-mcp entry, leaves every other server entry untouched.
hermes-memory install-mcp claude-desktop
# (or: cursor, cline)

# Enable embeddings in the MCP server too (default off so the first MCP
# call doesn't surprise-download 80 MB of ONNX weights):
export HERMES_MEMORY_EMBEDDINGS=1
```

After `install-mcp`, restart your client. The five tools appear in the tool list automatically.

## Why local-first

Every existing agent-memory MCP server I evaluated either (a) ships data to a third-party API, (b) requires a cloud index, or (c) requires extensive infra setup. None of those work for projects under NDA, on air-gapped networks, or for solo devs who don't want their notes leaving their laptop.

This server runs entirely on the user's machine. The index is a plain SQLite file. The corpus is the user's filesystem. The MCP transport is stdio. No network calls happen unless the user explicitly adds an embedding model that calls out (and a4 ships with a local default).

## Status milestones

- **v0.1.0a1** — package scaffold, tool surfaces, citation contract, 6 tests passing
- **v0.1.0a2** — official MCP SDK stdio loop, per-tool JSON schemas, install-mcp writers for Claude Desktop / Cursor / Cline, 21 tests passing
- **v0.1.0a3** — filesystem walker, SQLite FTS5 index, working `search_memory` + `find_decision` + `what_changed_since` tools + `hermes-memory init`/`ask` CLI, 49 tests passing
- **v0.1.0a4** — all 5 tools wired; YAML frontmatter parser; ADR status parser with reversal-chain awareness (`find_decision` now groups hits into "currently in force" vs "superseded/historical"); wiki/markdown link extractor; 69 tests passing
- **v0.1.0a5 (this release)** — opt-in semantic embeddings (FastEmbed / ONNX, no PyTorch) layered on FTS5 via Reciprocal Rank Fusion; `hermes-memory embed` CLI; `--embeddings` flag on `ask`; `HERMES_MEMORY_EMBEDDINGS=1` env switch for the MCP server; FTS-only path unchanged when the extra isn't installed; **101 tests passing**; PyPI publish
- **v0.1.0a5.1** — end-to-end demo against the [Claude Code Amnesia Test](https://github.com/nauta-ai/claude-code-amnesia-test) sample project, ≥22/25 score
- **v0.1.0** — stable, install-ready

## License

MIT
