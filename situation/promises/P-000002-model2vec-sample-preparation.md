## State

implemented

## Promise

With valid explicitly selected local inputs and a new or empty destination directory, the Model2Vec preparation surface produces a normalized, manifest-attested Rust sample for the benchmark experiment. TypeScript, Swift, and Kotlin inputs contribute only when explicitly enabled. Repeating preparation with the same inputs and configuration produces byte-identical outputs; evaluation-owned exact code or text is excluded from training outputs, available crate or repository groups do not straddle training and evaluation, and the Swift per-row grouping limitation is recorded. The surface rejects missing required Rust input, unsafe destinations, and dataset strings that cannot be safely parsed as mappings without executing them.

## Scope

This promise covers `benchmarks/model2vec_data/` and its focused tests in `tests/benchmarks/test_model2vec_adapters.py` and `tests/benchmarks/test_model2vec_prepare.py`. It applies only to the benchmark's explicitly selected local input files and output directory; it does not make a claim about remote dataset availability or content.

## Oracle

[O-000002](situation/oracles/O-000002-model2vec-sample-preparation.md)

## State evidence

- `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_data/adapters.py`, `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_data/prepare.py`, and `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_data/sources.py` are the implementation introduced by the trigger head.
- [D-000002](situation/decisions/D-000002-model2vec-benchmark-boundary.md) records the Rust-first benchmark boundary.

## Residual

This promise does not assure that a local input matches its public dataset revision, that optional sources are complete, any training or retrieval-quality outcome, production integration, or a fork gate claim.

## References

- `benchmarks/model2vec_data/README.md`