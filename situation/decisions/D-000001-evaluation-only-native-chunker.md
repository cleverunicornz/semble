## Status

accepted

## Date

2026-09-06

## Context

The BACKPORT opening trigger tree already contained the donor's native legacy cAST chunker comparison alongside Semble's existing chunking path: pre-opening commits `4a7b4bd442a0d6f3865568799b5cb1d92334ddb7` and `9d2d3e9ba4dc4f986b0aad3f2d51fff7b0dce1ec` introduced the comparison and its failing-support-check treatment before opening checkpoint `3f93c19ba404dd06c09e6fbbcb9740ded5d6bc1f`. The comparison needed a selected boundary that preserves production behavior while making an unusable native boundary visible to the evaluator.

## Evidence

- `4a7b4bd442a0d6f3865568799b5cb1d92334ddb7:benchmarks/chunker_eval.py` introduces the evaluation-only seam, current and legacy variants, isolated cache, and retained result payloads.
- `9d2d3e9ba4dc4f986b0aad3f2d51fff7b0dce1ec:benchmarks/chunker_eval.py` records the selected treatment of a failing native support check.
- `benchmarks/results/chunker-eval-smoke-current-fastapi.json` and `benchmarks/results/chunker-eval-smoke-legacy1500-fastapi.json` retain paired evaluation outputs.

## Decision

Keep the native legacy cAST chunker as an evaluation-only variant injected at the `semble.index.create.chunk_source` seam. Use Semble's current chunker for paths the native variant reports as unsupported, and surface native support or chunking failures as `ChunkerEvalError` rather than treating them as fallback results.

## Why

The selected seam permits comparable index and retrieval measurements while leaving production source and persistent cache formats unchanged. Explicit native-boundary errors keep an invalid comparison from appearing as a successful current-chunker result.

## Rejected alternatives

- Replacing Semble's production chunker with the native variant was not selected because the donor work is an evaluation, not a production integration.
- Treating a native support-check exception as an unsupported path was not selected because it would conceal a failed native boundary as a fallback measurement.

## Consequences

The comparison harness remains under `benchmarks/`; its result artifacts are observations for analysis, not proof of a production or fork-gate claim. A future production integration requires its own decision, promise, oracle, and witnesses.

## Revisit when

Revisit when the native chunker is proposed as a production dependency or its Python boundary changes.