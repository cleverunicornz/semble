## Identity

`semble` is the Clever Unicorn fork of a Python code-search package that produces a library, CLI, and context-bound MCP server for coding agents.

## Ownership

`UPSTREAM_FORK` — public upstream: https://github.com/MinishLab/semble.

## Phase

`EVOLUTION` — the opening tree already provides the source package, CLI entry point, MCP server, and regression suite; this closure adds repository knowledge rather than initial implementation.

## Implementation map

- `src/semble/` — package implementation; `index/`, `chunking/`, `search.py`, and `ranking/` cover index construction and retrieval.
- `src/semble/cli.py` and `src/semble/mcp.py` — command-line and context-bound MCP surfaces.
- `src/semble/git_workspace.py` and `src/semble/workspace.py` — Git-bound immutable-baseline and changed-workspace service.
- `tests/` — regression coverage; `.github/workflows/ci.yaml` is inherited workflow evidence, not an assured fork gate route.

## Current state

The repository retains upstream-owned source, documentation, configuration, and workflows unchanged. Fork orientation is recorded only in root `AGENTS.md` and `situation/`.

## Intended state

No fork-specific behavior or assurance commitment is created by this BACKPORT. Future fork changes must establish their own records from observed evidence.

## Closure state

- Current run: `20260906T195316Z-acecfcfcb88c2ba4a4f47c74a2e68abed0baf0ce` (open)
- Last completed closure: none
- Transcript: none
