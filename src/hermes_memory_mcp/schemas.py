"""Per-tool JSON Schemas for the 5 Hermes Memory MCP tools.

These schemas are what the MCP client (Claude Desktop, Cursor, Cline, etc.)
sees during tool discovery. They define the contract the agent must follow
when calling each tool.

v0.1.0a2: schemas are minimal but spec-correct — every input is validated by
the MCP SDK before reaching the tool function, so we don't need to re-check
required fields inside the implementations.
"""

from __future__ import annotations

# JSON Schema fragments. Keep them small + descriptive: agents read these.

SEARCH_MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Free-text query. Treat it as natural language, not a regex "
                "or keyword expression."
            ),
        },
        "scope": {
            "type": "string",
            "enum": ["all", "notes", "decisions", "logs", "code", "git"],
            "default": "all",
            "description": (
                "Restrict the search to one corpus. 'notes' = Markdown notes "
                "+ Obsidian vault. 'decisions' = ADRs only. 'logs' = daily "
                "log files. 'code' = source files. 'git' = commit messages."
            ),
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 10,
            "description": "Max number of ranked passages to return.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

GET_PROJECT_BRIEF_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "repo_or_topic": {
            "type": "string",
            "default": "current",
            "description": (
                "Project, repo, or topic name. 'current' = the project the "
                "agent appears to be working in based on cwd / open files."
            ),
        },
        "as_of": {
            "type": "string",
            "default": "now",
            "description": (
                "Time anchor. 'now', a date 'YYYY-MM-DD', or a named "
                "snapshot id. Briefs are compiled deterministically from "
                "indexed memory, not LLM-generated."
            ),
        },
    },
    "additionalProperties": False,
}

FIND_DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": (
                "What was decided about. Returns the ADR (if any), the full "
                "reversal chain, and the current effective decision."
            ),
        },
    },
    "required": ["topic"],
    "additionalProperties": False,
}

WHAT_CHANGED_SINCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reference": {
            "type": "string",
            "description": (
                "What to diff against. Accepts 'last_session', an ISO date "
                "'YYYY-MM-DD', or a named snapshot id."
            ),
        },
    },
    "required": ["reference"],
    "additionalProperties": False,
}

CHECK_CLAIM_AGAINST_MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claim": {
            "type": "string",
            "description": (
                "A factual claim the agent is about to make or act on. "
                "Returns a verdict (supported / contradicted / unknown) "
                "with citations to the strongest contradicting / supporting "
                "sources from the indexed corpus."
            ),
        },
    },
    "required": ["claim"],
    "additionalProperties": False,
}


TOOL_SCHEMAS: dict[str, dict] = {
    "search_memory": SEARCH_MEMORY_SCHEMA,
    "get_project_brief": GET_PROJECT_BRIEF_SCHEMA,
    "find_decision": FIND_DECISION_SCHEMA,
    "what_changed_since": WHAT_CHANGED_SINCE_SCHEMA,
    "check_claim_against_memory": CHECK_CLAIM_AGAINST_MEMORY_SCHEMA,
}


TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_memory": (
        "Free-text semantic search across the user's memory corpus (notes, "
        "decisions, logs, code, git). Returns ranked passages with file_path "
        "+ line_range citations. Use this before answering any factual "
        "question about the project."
    ),
    "get_project_brief": (
        "Return a compiled current-state brief on a project, repo, or topic. "
        "Briefs are deterministic summaries built from indexed memory, NOT "
        "LLM-generated paraphrase. Use this when a session starts cold and "
        "you need orientation."
    ),
    "find_decision": (
        "Return the decision chain on a topic: original ADR, every reversal, "
        "and the current effective decision. Use this before re-litigating "
        "anything that smells like 'we already decided this'."
    ),
    "what_changed_since": (
        "Diff the current memory snapshot against a prior reference "
        "(last_session, a date, or a named snapshot). Returns new/modified "
        "entries grouped by doc_type. Use this to catch up after time away."
    ),
    "check_claim_against_memory": (
        "Cassandra-style contradiction check: given a claim, surface "
        "contradicting sources and a verdict (supported / contradicted / "
        "unknown). Use this before committing to any 'X is true' assertion."
    ),
}
