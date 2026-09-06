## State

implemented

## Promise

With valid arguments and available selected benchmark inputs, the evaluation-only chunker harness evaluates Semble's current chunking path or the native legacy cAST chunking path at the index chunk-source seam, emits a labeled JSON measurement, and rejects native-boundary failures rather than presenting them as fallback measurements.

## Scope

This promise covers `benchmarks/chunker_eval.py` in its `current`, `legacy-cast`, and `chunk-timing` modes, including its temporary evaluation cache and `benchmarks/results/` output. It covers only the harness's own process and selected benchmark inputs.

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