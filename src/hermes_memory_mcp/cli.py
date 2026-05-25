"""hermes-memory CLI — operator-facing commands.

Subcommands (v0.1.0a1 placeholders; full impl in subsequent sessions):
    init           — point at a project root, build the local memory index
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
from .cite import Citation, CitedResponse
from .index import Index
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
    with Index.open(root) as ix:
        hits = ix.search(args.query, scope=args.scope, limit=args.limit)
    if not hits:
        print(f"No results for: {args.query!r}")
        return 0
    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] {hit.file_path}  ({hit.doc_type}, rank={hit.rank:.2f})")
        # Snippet may contain newlines from FTS5; flatten for one-line preview.
        snippet = " ".join(hit.snippet.split())
        print(f"    {snippet}")
    return 0


def cmd_install_mcp(args: argparse.Namespace) -> int:
    command = args.command_path or install_mod.detect_command()
    try:
        path = install_mod.install(args.client, command=command)
    except Exception as exc:  # noqa: BLE001 — surface to the operator verbatim
        print(f"install-mcp failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.client} config: {path}")
    print(f"MCP server command: {command}")
    print("Restart the client to pick up the new server.")
    return 0


def cmd_uninstall_mcp(args: argparse.Namespace) -> int:
    try:
        path, removed = install_mod.uninstall(args.client)
    except Exception as exc:  # noqa: BLE001
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
    if args.command == "install-mcp":
        return cmd_install_mcp(args)
    if args.command == "uninstall-mcp":
        return cmd_uninstall_mcp(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
