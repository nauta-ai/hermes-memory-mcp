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

from . import __version__


def cmd_version() -> int:
    print(f"hermes-memory {__version__}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    print(f"v0.1.0a1 stub: would index {args.root_path!r}")
    print("Next session: implement filesystem walker + SQLite vector index build.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    print(f"v0.1.0a1 stub: would ask {args.query!r}")
    return 0


def cmd_install_mcp(args: argparse.Namespace) -> int:
    print(f"v0.1.0a1 stub: would install MCP server into {args.client!r} config")
    print("Next session: implement claude_desktop_config.json + cursor mcp_settings.json writers.")
    return 0


def cmd_uninstall_mcp(args: argparse.Namespace) -> int:
    print(f"v0.1.0a1 stub: would remove MCP server entry from {args.client!r} config")
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

    inst = sub.add_parser("install-mcp", help="Wire MCP server into a client")
    inst.add_argument("client", choices=["claude-desktop", "cursor", "cline"])

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
