"""Filesystem walker — yields indexable documents from a project root.

The walker is the *only* component that touches disk. Everything downstream
(parsers, index, search) consumes its output. Keeping the boundary thin
makes it easy to swap walker backends later (e.g. git-aware walker for
"only files known to git", or a remote-backed walker for shared corpora).

Doc-type classification happens here because it determines the parser
chosen later. We use file path + extension only — no content sniffing —
so the walker stays cheap to re-run.

Default ignores: anything matching .gitignore-style patterns commonly seen
in source repos. Users can override via the ``extra_ignore`` argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Doc types map to parser dispatch in parsers.py. Keep in sync.
DOC_TYPE_MARKDOWN = "markdown"
DOC_TYPE_ADR = "adr"
DOC_TYPE_CODE = "code"
DOC_TYPE_GIT = "git"
DOC_TYPE_LOG = "log"
DOC_TYPE_OTHER = "other"

# Files / dirs we never want indexed. Mirrors common .gitignore intent
# without trying to parse the actual .gitignore — for a3 this is enough,
# a4 wires the real .gitignore parser if users complain.
DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "target",  # rust build dir
        "build",
        "dist",
        ".next",
        ".svelte-kit",
        ".cache",
        ".idea",
        ".vscode",
    }
)

# Extensions we treat as indexable code. Anything else (binaries, images,
# data files) is skipped silently.
CODE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi",
        ".rs",
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
        ".go",
        ".java", ".kt", ".scala",
        ".rb",
        ".c", ".cc", ".cpp", ".h", ".hpp",
        ".swift",
        ".sh", ".bash", ".zsh",
        ".sql",
        ".toml", ".yaml", ".yml", ".json",
        ".html", ".css", ".scss",
        ".svelte", ".vue",
    }
)

MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown", ".mdx"})

LOG_HINTS = frozenset({".log"})

# An ADR file is a markdown file that lives under a directory literally
# named "adr", "adrs", "decisions", or "doc/adr"-shaped paths. Heuristic
# only — users can supplement via metadata in a4.
ADR_DIR_HINTS = frozenset({"adr", "adrs", "decisions"})


@dataclass(frozen=True)
class Document:
    """One indexable unit from the corpus.

    file_path  — absolute path on disk
    doc_type   — one of the DOC_TYPE_* constants above
    content    — full file text (UTF-8, decoded with errors="replace")
    mtime      — file modification time (Unix epoch seconds)
    size       — content length in bytes (post-decode)
    """

    file_path: Path
    doc_type: str
    content: str
    mtime: float
    size: int


def classify(path: Path) -> str:
    """Map a path to a doc_type. Pure function — no I/O."""
    name = path.name.lower()
    ext = path.suffix.lower()

    if name in {"changelog.md", "changelog"} and ext in MARKDOWN_EXTENSIONS | {""}:
        return DOC_TYPE_MARKDOWN

    if ext in MARKDOWN_EXTENSIONS:
        # ADR heuristic: any markdown file under a directory whose name is
        # in ADR_DIR_HINTS, anywhere in its ancestry within the project.
        for parent in path.parents:
            if parent.name.lower() in ADR_DIR_HINTS:
                return DOC_TYPE_ADR
        return DOC_TYPE_MARKDOWN

    if ext in LOG_HINTS:
        return DOC_TYPE_LOG

    if ext in CODE_EXTENSIONS:
        return DOC_TYPE_CODE

    return DOC_TYPE_OTHER


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8. Returns None for unreadable files (binary,
    permission denied, etc.) — caller skips silently."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None


def walk(
    root: Path,
    *,
    extra_ignore: frozenset[str] = frozenset(),
    max_file_size: int = 1_000_000,
    include_other: bool = False,
) -> Iterator[Document]:
    """Yield Documents under ``root``.

    Skips ignored directories, files larger than ``max_file_size`` bytes
    (default 1 MB — a single source file rarely exceeds this; bigger files
    are usually data dumps or build artifacts), and unreadable files.

    By default, files classified as DOC_TYPE_OTHER are skipped. Pass
    ``include_other=True`` to yield them anyway (useful for completeness
    audits, not normal indexing).
    """
    root = root.expanduser().resolve()
    ignore_dirs = DEFAULT_IGNORE_DIRS | extra_ignore

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for filename in filenames:
            path = Path(dirpath) / filename

            try:
                stat = path.stat()
            except OSError:
                continue

            if stat.st_size > max_file_size:
                continue

            doc_type = classify(path)
            if doc_type == DOC_TYPE_OTHER and not include_other:
                continue

            content = _read_text(path)
            if content is None:
                continue

            yield Document(
                file_path=path,
                doc_type=doc_type,
                content=content,
                mtime=stat.st_mtime,
                size=len(content),
            )
