"""The citation contract is a non-negotiable invariant of this package.

This test exists to ensure any code path that returns a non-empty response
also carries at least one citation. If it ever passes a no-citation response
that has content, the test fails — guarding against regression of the
"differentiator from RAG" design call.
"""

from __future__ import annotations

import pytest

from hermes_memory_mcp.cite import Citation, CitedResponse, empty_result


def test_empty_response_with_no_citations_is_legal():
    """Pure no-content response (empty corpus, no results) is allowed."""
    r = CitedResponse(content="", citations=[])
    assert r.content == ""
    assert r.citations == []


def test_non_empty_response_without_citations_raises():
    """The contract: any content → at least one citation."""
    with pytest.raises(ValueError, match="citation"):
        CitedResponse(content="Here is an answer.", citations=[])


def test_response_with_citation_serializes_to_expected_shape():
    cit = Citation(
        file_path="decisions/0007-rate-limit.md",
        line_range="12-18",
        snippet="Rate-limiting per tenant prevents noisy-neighbor degradation.",
    )
    r = CitedResponse(content="Rate-limit per tenant per ADR 0007.", citations=[cit])
    payload = r.to_dict()
    assert payload == {
        "content": "Rate-limit per tenant per ADR 0007.",
        "citations": [
            {
                "file_path": "decisions/0007-rate-limit.md",
                "line_range": "12-18",
                "snippet": "Rate-limiting per tenant prevents noisy-neighbor degradation.",
            }
        ],
    }


def test_citation_rejects_empty_file_path():
    with pytest.raises(ValueError, match="file_path"):
        Citation(file_path="")


def test_empty_result_helper_bypasses_contract():
    """The 'no results' path explicitly uses empty_result() rather than constructing
    a bare CitedResponse, so the bypass is intentional + auditable."""
    r = empty_result("nothing matched query 'frobnicate'")
    assert r.citations == []
    assert "No results" in r.content
    assert "frobnicate" in r.content


def test_response_with_multiple_citations():
    cits = [
        Citation(file_path="decisions/0004-microservices.md", line_range="1-5"),
        Citation(file_path="decisions/0005-monolith-instead.md", line_range="3-9"),
    ]
    r = CitedResponse(content="ADR 0004 reversed by ADR 0005.", citations=cits)
    assert len(r.citations) == 2
    assert all(c.file_path for c in r.citations)
