## Identity

`semble` is the Clever Unicorn fork of a Python code-search package that produces a library, CLI, and context-bound MCP server for coding agents.

## Ownership

`UPSTREAM_FORK` — public upstream: https://github.com/MinishLab/semble.

## Phase

`INITIAL` — this is the first situation context on this branch; the opening tree already contains the package implementation, and this BACKPORT begins its repository-knowledge projection.

## Implementation map

- `src/semble/` — package implementation for indexing, chunking, ranking, search, and workspace handling.
- `src/semble/cli.py` and `src/semble/mcp.py` — command-line and context-bound MCP surfaces.
- `src/semble/git_workspace.py` and `src/semble/workspace.py` — Git-bound baseline and changed-workspace services.
- `benchmarks/` — benchmark fixtures, evaluation scripts, and retained result artifacts, including the evaluation-only native chunker comparison at `benchmarks/chunker_eval.py`.
- `tests/` — regression suite; `.github/workflows/ci.yaml` is inherited workflow evidence, not an assured fork gate route.

## Current state

The repository retains upstream-owned source, documentation, configuration, and workflows unchanged by this closure. Fork orientation lives only in root `AGENTS.md` and `situation/`.

## Intended state

This BACKPORT records existing repository knowledge without creating a fork-specific production behavior or gate-assurance commitment.

## Closure state

- Current run: none
- Last completed closure: run `20260906T211928Z-218ec16464c0290c6567e4cb29c3532d594bf75c`, opened at `1e6c4aa506cdd7843dddb19a2fe519480fc986f3`
- Transcript: `s3://cvu-automation-runs-uk/bedrock/cleverunicornz/semble/pr-11/20260906T211928Z-218ec16464c0290c6567e4cb29c3532d594bf75c/`
