"""install-mcp / uninstall-mcp config writers.

Each MCP client (Claude Desktop, Cursor, Cline) stores its server registry
in a JSON config file. Their schemas overlap but aren't identical, so each
writer is its own function. All three share the same pattern:

    1. Read the existing config (creating an empty one if missing).
    2. Splice in (or remove) the ``hermes-memory-mcp`` entry, leaving every
       other server registration untouched.
    3. Write the file back atomically.

We never overwrite the whole config. Users may have other MCP servers
already registered; trampling those would be a hostile install.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SERVER_KEY = "hermes-memory-mcp"


@dataclass(frozen=True)
class ClientTarget:
    """One MCP client's config location + shape."""

    name: str
    config_path: Path
    # The JSON path (list of keys) under which MCP servers are registered.
    # Claude Desktop:  config["mcpServers"][SERVER_KEY] = entry
    # Cursor:          config["mcpServers"][SERVER_KEY] = entry
    # Cline:           config["mcpServers"][SERVER_KEY] = entry
    # All three use the same "mcpServers" object today.
    servers_key: str = "mcpServers"


def _claude_desktop_config_path() -> Path:
    """Return Claude Desktop's config path for the current OS.

    macOS:    ~/Library/Application Support/Claude/claude_desktop_config.json
    Linux:    ~/.config/Claude/claude_desktop_config.json
    Windows:  %APPDATA%\\Claude\\claude_desktop_config.json
    """
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA env var not set; cannot locate Claude Desktop config")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    # Linux / other Unix
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def _cursor_config_path() -> Path:
    """Cursor stores MCP config at ~/.cursor/mcp.json (project-scoped configs
    live at <project>/.cursor/mcp.json — we target the global one)."""
    return Path.home() / ".cursor" / "mcp.json"


def _cline_config_path() -> Path:
    """Cline stores MCP config under the VS Code globalStorage path.

    Path under each OS root:
      User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
    """
    sub = (
        Path("User")
        / "globalStorage"
        / "saoudrizwan.claude-dev"
        / "settings"
        / "cline_mcp_settings.json"
    )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / sub
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA env var not set; cannot locate Cline config")
        return Path(appdata) / "Code" / sub
    return Path.home() / ".config" / "Code" / sub


CLIENTS: dict[str, ClientTarget] = {
    "claude-desktop": ClientTarget("claude-desktop", _claude_desktop_config_path()),
    "cursor": ClientTarget("cursor", _cursor_config_path()),
    "cline": ClientTarget("cline", _cline_config_path()),
}


def _server_entry(command: str | None = None) -> dict:
    """The MCP server entry we splice into client configs.

    ``command`` defaults to the ``hermes-memory-mcp`` console script on PATH.
    Callers can pin to an absolute path (e.g. inside a venv) for stability.
    """
    return {
        "command": command or "hermes-memory-mcp",
        "args": [],
    }


def _read_config(path: Path) -> dict:
    """Read a JSON config, returning {} if it doesn't exist yet. Raises on
    malformed JSON so we don't silently overwrite a corrupt file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Refusing to touch {path}: existing file is not valid JSON ({exc}). "
            "Fix or remove it manually, then re-run install-mcp."
        ) from exc


def _write_config_atomic(path: Path, data: dict) -> None:
    """Write JSON atomically: tmp file in the same dir, then rename. This
    survives a crash mid-write without corrupting the user's config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def install(client: str, command: str | None = None) -> Path:
    """Splice the hermes-memory-mcp entry into ``client``'s config.

    Returns the config path that was written. Leaves every other server
    entry untouched. Idempotent — re-running just refreshes our entry.
    """
    target = CLIENTS[client]
    config = _read_config(target.config_path)
    servers = config.setdefault(target.servers_key, {})
    servers[SERVER_KEY] = _server_entry(command)
    _write_config_atomic(target.config_path, config)
    return target.config_path


def uninstall(client: str) -> tuple[Path, bool]:
    """Remove the hermes-memory-mcp entry from ``client``'s config.

    Returns (config_path, removed). ``removed`` is False if the entry
    wasn't present (so callers can report "nothing to do" cleanly).
    Other server entries are left untouched.
    """
    target = CLIENTS[client]
    if not target.config_path.exists():
        return target.config_path, False
    config = _read_config(target.config_path)
    servers = config.get(target.servers_key, {})
    if SERVER_KEY not in servers:
        return target.config_path, False
    del servers[SERVER_KEY]
    _write_config_atomic(target.config_path, config)
    return target.config_path, True


def detect_command() -> str:
    """Find the absolute path to ``hermes-memory-mcp`` on PATH.

    Pinning to an absolute path makes the install survive PATH changes
    (e.g. after a shell config rewrite). Falls back to the bare command
    name if the binary isn't on PATH yet — useful in dev installs where
    the user will fix PATH later.
    """
    found = shutil.which("hermes-memory-mcp")
    return found or "hermes-memory-mcp"
