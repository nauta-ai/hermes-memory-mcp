"""Citation contract enforcement.

Every tool response MUST carry source citations. This module is the single
source of truth for that contract — all 5 tool responses route through
``format_with_citations`` before returning. A response with no sources is
a bug, not a feature, and ``test_citation_required.py`` enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    """A single source citation backing some part of a response.

    file_path  — absolute or repo-relative path to the source
    line_range — string like '12-18' or '47' (line number(s))
    snippet    — verbatim text from the source (max ~200 chars), or None
                 if the citation is a whole-file reference
    """

    file_path: str
    line_range: str | None = None
    snippet: str | None = None

    def __post_init__(self) -> None:
        if not self.file_path or not self.file_path.strip():
            raise ValueError("Citation.file_path must be non-empty")


@dataclass
class CitedResponse:
    """A response payload that carries citations.

    ``content`` is the rendered answer (markdown). ``citations`` is the
    list of sources. The contract: ``len(citations) >= 1`` for any tool
    response that produces a non-trivial answer. The single exception
    is an explicit "no results" return, which must use the
    ``empty_result`` helper rather than constructing a bare CitedResponse.
    """

    content: str
    citations: list[Citation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.citations, list):
            raise TypeError("CitedResponse.citations must be a list")
        if self.content and not self.citations:
            raise ValueError(
                "CitedResponse with non-empty content must carry at least one "
                "citation. Use empty_result() for the 'no results' path."
            )

    def to_dict(self) -> dict:
        """Serialize for MCP tool response. Stable schema; agents consume this."""
        return {
            "content": self.content,
            "citations": [
                {
                    "file_path": c.file_path,
                    "line_range": c.line_range,
                    "snippet": c.snippet,
                }
                for c in self.citations
            ],
        }


def empty_result(reason: str) -> CitedResponse:
    """Construct an empty response with a reason. Bypasses the citation
    requirement because there is genuinely nothing to cite.

    Implementation note: uses object.__setattr__ to set ``content`` AFTER
    __post_init__ runs, so the citation-required check (which fires when
    content is non-empty) doesn't trigger. The bypass is intentional + only
    happens through this helper, so every "no citations" case is auditable
    by grepping for empty_result() at call sites."""
    r = CitedResponse(content="", citations=[])
    object.__setattr__(r, "content", f"No results: {reason}")
    return r
