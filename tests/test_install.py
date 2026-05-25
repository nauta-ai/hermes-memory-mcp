"""Tests for install-mcp / uninstall-mcp config writers.

These tests must NOT touch real Claude Desktop / Cursor / Cline configs.
Every test patches the client target's config_path to a tmp_path fixture
so a failed install never corrupts a real user config.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_memory_mcp import install as install_mod


@pytest.fixture
def fake_config_path(tmp_path: Path) -> Path:
    return tmp_path / "client_config.json"


def _patched_clients(fake_path: Path) -> dict:
    """Build a CLIENTS-shaped dict where every client points at fake_path,
    so we never write to real config files during tests."""
    return {
        name: install_mod.ClientTarget(name=name, config_path=fake_path)
        for name in ("claude-desktop", "cursor", "cline")
    }


def test_install_creates_new_config_with_entry(fake_config_path: Path) -> None:
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        path = install_mod.install("claude-desktop", command="/usr/local/bin/hermes-memory-mcp")
    assert path == fake_config_path
    data = json.loads(fake_config_path.read_text())
    assert "mcpServers" in data
    assert data["mcpServers"]["hermes-memory-mcp"] == {
        "command": "/usr/local/bin/hermes-memory-mcp",
        "args": [],
    }


def test_install_preserves_other_server_entries(fake_config_path: Path) -> None:
    """User may already have other MCP servers registered. Install must
    splice in our entry without touching theirs."""
    existing = {
        "mcpServers": {
            "filesystem": {"command": "mcp-filesystem", "args": ["/home/me"]},
            "github": {"command": "mcp-github", "args": []},
        },
        "otherTopLevelKey": "must-survive",
    }
    fake_config_path.write_text(json.dumps(existing))
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        install_mod.install("cursor", command="hermes-memory-mcp")
    data = json.loads(fake_config_path.read_text())
    assert data["otherTopLevelKey"] == "must-survive"
    assert set(data["mcpServers"].keys()) == {"filesystem", "github", "hermes-memory-mcp"}
    assert data["mcpServers"]["filesystem"]["command"] == "mcp-filesystem"


def test_install_is_idempotent(fake_config_path: Path) -> None:
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        install_mod.install("cline", command="hermes-memory-mcp")
        install_mod.install("cline", command="hermes-memory-mcp")
    data = json.loads(fake_config_path.read_text())
    assert list(data["mcpServers"].keys()) == ["hermes-memory-mcp"]


def test_uninstall_removes_only_our_entry(fake_config_path: Path) -> None:
    existing = {
        "mcpServers": {
            "filesystem": {"command": "mcp-filesystem", "args": []},
            "hermes-memory-mcp": {"command": "hermes-memory-mcp", "args": []},
        },
    }
    fake_config_path.write_text(json.dumps(existing))
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        path, removed = install_mod.uninstall("claude-desktop")
    assert removed is True
    assert path == fake_config_path
    data = json.loads(fake_config_path.read_text())
    assert "hermes-memory-mcp" not in data["mcpServers"]
    assert "filesystem" in data["mcpServers"]


def test_uninstall_missing_entry_is_noop(fake_config_path: Path) -> None:
    fake_config_path.write_text(json.dumps({"mcpServers": {}}))
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        path, removed = install_mod.uninstall("cursor")
    assert removed is False


def test_uninstall_missing_config_is_noop(fake_config_path: Path) -> None:
    # File doesn't exist
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        path, removed = install_mod.uninstall("cline")
    assert removed is False
    assert not fake_config_path.exists()


def test_install_rejects_malformed_existing_config(fake_config_path: Path) -> None:
    fake_config_path.write_text("{ not valid json")
    with patch.object(install_mod, "CLIENTS", _patched_clients(fake_config_path)):
        with pytest.raises(RuntimeError, match="not valid JSON"):
            install_mod.install("claude-desktop")


def test_atomic_write_does_not_leave_partial_file(tmp_path: Path) -> None:
    """The atomic write must never leave a half-written .json behind, even
    if the rename target already exists."""
    target = tmp_path / "exists.json"
    target.write_text('{"old": true}')
    install_mod._write_config_atomic(target, {"new": True})
    data = json.loads(target.read_text())
    assert data == {"new": True}
    # No stragglers from tempfile
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == [], f"Atomic write left stragglers: {leftovers}"
