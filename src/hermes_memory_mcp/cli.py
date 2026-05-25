"""hermes-memory CLI — operator-facing commands.

Subcommands:
    init           — point at a project root, build the local memory index
    embed          — populate semantic embeddings for the local index (a5)
    ask            — interactive query against the index (no MCP client needed)
    install-mcp    — wire the MCP server into Claude Desktop / Cursor / Cline
    uninstall-mcp  — reverse of install-mcp
    version        — print version

This CLI is the developer-facing surface. The MCP server proper is
``hermes-memory-mcp`` (a separate console_script). They share the same
codebase but expose different interfaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import install as install_mod
from .index import Index
from .search import hybrid_search
from .walker import walk


def cmd_version() -> int:
    print(f"hermes-memory {__version__}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root_path).expanduser().resolve()
    if not root.is_dir():
        print(f"init: {root} is not a directory", file=sys.stderr)
        return 1
    print(f"Indexing {root}...")
    with Index.open(root) as ix:
        n = ix.add_many(walk(root))
        breakdown = ix.doc_types()
    print(f"Indexed {n} documents at {ix.db_path}")
    if breakdown:
        for doc_type, count in sorted(breakdown.items()):
            print(f"  {doc_type}: {count}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    root = Path(args.root or ".").expanduser().resolve()
    embedder = None
    if args.embeddings:
        try:
            from .embedder import EmbedderUnavailableError, get_default

            embedder = get_default()
        except EmbedderUnavailableError as exc:
            print(f"ask: {exc}", file=sys.stderr)
            return 1
    with Index.open(root) as ix:
        hits = hybrid_search(ix, args.query, scope=args.scope, limit=args.limit, embedder=embedder)
    if not hits:
        print(f"No results for: {args.query!r}")
        return 0
    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] {hit.file_path}  ({hit.doc_type}, rank={hit.rank:.4f})")
        # Snippet may contain newlines from FTS5; flatten for one-line preview.
        snippet = " ".join(hit.snippet.split())
        print(f"    {snippet}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    """Populate embeddings for every doc in the index that doesn't have
    one yet. Safe to re-run — only the NULL-embedding docs are touched."""
    root = Path(args.root or ".").expanduser().resolve()
    try:
        from .embedder import EmbedderUnavailableError, get_default
    except ImportError:
        print(
            "embed: fastembed is not installed.\n  pip install 'hermes-memory-mcp[embeddings]'",
            file=sys.stderr,
        )
        return 1
    try:
        embedder = get_default()
    except EmbedderUnavailableError as exc:
        print(f"embed: {exc}", file=sys.stderr)
        return 1
    with Index.open(root) as ix:
        before_embedded, total = ix.embedding_coverage()
        if total == 0:
            print(f"No documents in {ix.db_path} — run `hermes-memory init {root}` first.")
            return 1
        print(f"Embedding pending documents using {embedder.model_name} (dim={embedder.dim})...")
        added = ix.embed_all_pending(embedder, batch_size=args.batch_size)
        after_embedded, total_after = ix.embedding_coverage()
    print(f"Embedded {added} documents this run.")
    print(f"Coverage: {after_embedded}/{total_after} (was {before_embedded}/{total}).")
    return 0


def cmd_install_mcp(args: argparse.Namespace) -> int:
    command = args.command_path or install_mod.detect_command()
    try:
        path = install_mod.install(args.client, command=command)
    except Exception as exc:
        print(f"install-mcp failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.client} config: {path}")
    print(f"MCP server command: {command}")
    print("Restart the client to pick up the new server.")
    return 0


def cmd_uninstall_mcp(args: argparse.Namespace) -> int:
    try:
        path, removed = install_mod.uninstall(args.client)
    except Exception as exc:
        print(f"uninstall-mcp failed: {exc}", file=sys.stderr)
        return 1
    if not removed:
        print(f"Nothing to remove from {args.client} config ({path}).")
        return 0
    print(f"Removed hermes-memory-mcp from {args.client} config: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-memory",
        description="Local-first memory MCP for AI coding agents — operator CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("version", help="Print version")

    init = sub.add_parser("init", help="Build the local memory index")
    init.add_argument("root_path", help="Project root to index (e.g. ~/projects/my-app)")

    ask = sub.add_parser("ask", help="Query the index interactively (no MCP client needed)")
    ask.add_argument("query", help="The question to ask")
    ask.add_argument(
        "--root",
        default=".",
        help="Project root whose index to query (default: current directory)",
    )
    ask.add_argument(
        "--scope",
        default="all",
        choices=["all", "markdown", "code", "adr", "log", "git"],
        help="Restrict the search to one doc_type",
    )
    ask.add_argument("--limit", type=int, default=10, help="Max results to show")
    ask.add_argument(
        "--embeddings",
        action="store_true",
        help="Blend FTS5 with semantic vector search via RRF. Requires the "
        "[embeddings] extra to be installed.",
    )

    emb = sub.add_parser("embed", help="Populate semantic embeddings for the index")
    emb.add_argument(
        "--root",
        default=".",
        help="Project root whose index to embed (default: current directory)",
    )
    emb.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Documents to encode per FastEmbed call (default: 32)",
    )

    inst = sub.add_parser("install-mcp", help="Wire MCP server into a client")
    inst.add_argument("client", choices=["claude-desktop", "cursor", "cline"])
    inst.add_argument(
        "--command-path",
        default=None,
        help=(
            "Absolute path to the hermes-memory-mcp binary. Defaults to the "
            "one found on PATH; pin this for venv-based installs."
        ),
    )

    uninst = sub.add_parser("uninstall-mcp", help="Remove MCP entry from a client")
    uninst.add_argument("client", choices=["claude-desktop", "cursor", "cline"])

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None or args.command == "version":
        return cmd_version()
    if args.command == "init":
        return cmd_init(args)
    if args.command == "ask":
        return cmd_ask(args)
    if args.command == "embed":
        return cmd_embed(args)
    if args.command == "install-mcp":
        return cmd_install_mcp(args)
    if args.command == "uninstall-mcp":
        return cmd_uninstall_mcp(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
