## Identity

`semble` is the Clever Unicorn fork of a Python code-search package that produces a library, CLI, and context-bound MCP server for coding agents.

## Ownership

`UPSTREAM_FORK` — public upstream: https://github.com/MinishLab/semble.

## Phase

`INITIAL` — this DELTA retains the repository's declared initial phase while recording a benchmark-only Model2Vec capability alongside the existing package behavior.

## Implementation map

- `src/semble/` — package implementation for indexing, chunking, ranking, search, and workspace handling.
- `src/semble/cli.py` and `src/semble/mcp.py` — command-line and context-bound MCP surfaces.
- `src/semble/git_workspace.py` and `src/semble/workspace.py` — Git-bound baseline and changed-workspace services.
- `benchmarks/` — benchmark fixtures, Model2Vec sample-preparation and smoke/full training-evaluation surfaces, and retained result artifacts, including the evaluation-only native chunker comparison at `benchmarks/chunker_eval.py`.
- `tests/` — regression suite; `.github/workflows/ci.yaml` is inherited workflow evidence, not an assured fork gate route.

## Current state

The current branch includes benchmark-only Model2Vec data preparation, smoke/full training, evaluation, and retained receipt surfaces. Fork orientation lives only in root `AGENTS.md` and `situation/`.

## Intended state

This DELTA records the Model2Vec benchmark behavior and evidence without creating a production Model2Vec deployment or a fork gate-assurance commitment.

## Closure state

- Current run: `20260907T143133Z-42fc95450b74b2f96e37c931dc16b5441319cafd` (open)
- Last completed closure: run `20260906T211928Z-218ec16464c0290c6567e4cb29c3532d594bf75c`, opened at `1e6c4aa506cdd7843dddb19a2fe519480fc986f3`
- Transcript: `s3://cvu-automation-runs-uk/bedrock/cleverunicornz/semble/pr-11/20260906T211928Z-218ec16464c0290c6567e4cb29c3532d594bf75c/`
