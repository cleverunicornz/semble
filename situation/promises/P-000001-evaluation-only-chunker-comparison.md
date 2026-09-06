## State

implemented

## Promise

With valid arguments and available selected benchmark inputs, the evaluation-only chunker harness provides the following mode-specific measurements:

- In `current`, it runs a retrieval comparison through Semble's existing chunk-source function at the index chunk-source seam. A successful run writes labeled JSON identifying `current`.
- In `legacy-cast`, it runs a retrieval comparison at that seam; it calls native `chunk_source` only for paths accepted by `supports_path`, uses Semble's current chunker for paths the predicate rejects, preserves the selected path and returned line metadata in converted Semble chunks, and treats a native support-check or chunking exception as an evaluation error rather than a successful fallback measurement. A successful run writes labeled JSON identifying `legacy-cast`.
- In `chunk-timing`, it independently measures the native `chunk_files` batch API rather than a retrieval comparison, accepts only a list with one result for each expected path in the same order and a stable file/chunk/error shape across repetitions, and records the count of returned per-file errors. A successful run writes labeled JSON identifying `chunk-timing`.

## Scope

This promise covers `benchmarks/chunker_eval.py` in its `current` and `legacy-cast` retrieval modes at the `semble.index.create.chunk_source` seam and in its independent `chunk-timing` native batch mode. It includes the retrieval modes' temporary evaluation cache and every mode's labeled JSON output under `benchmarks/results/`, and covers only the harness process and selected benchmark inputs.

## Oracle

[O-000001](situation/oracles/O-000001-evaluation-only-chunker-comparison.md)

## State evidence

- `4a7b4bd442a0d6f3865568799b5cb1d92334ddb7:benchmarks/chunker_eval.py` is the implementation commit.
- [D-000001](situation/decisions/D-000001-evaluation-only-native-chunker.md) records the selected native-boundary behavior.

## Residual

This promise does not assure metric quality, native-module availability or semantics, behavior outside selected benchmark inputs, production integration, or any fork gate claim.

## References

- `benchmarks/results/chunker-eval-smoke-current-fastapi.json`
- `benchmarks/results/chunker-eval-smoke-legacy1500-fastapi.json`