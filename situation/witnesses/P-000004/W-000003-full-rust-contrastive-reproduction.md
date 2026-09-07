## Promise

[P-000004](situation/promises/P-000004-full-rust-contrastive-evaluation.md)

## Oracle

[O-000004](situation/oracles/O-000004-full-rust-contrastive-evaluation.md)

## Result

INVALID

## Head

`f586c664ae121da213910daabf7d8b3dd1ea81b2`

## Observed

2026-09-07

## Evidence

- `benchmarks/model2vec_training/receipts/massed-20260907-1b2adaac.yml` is the committed result file for reproduction run `rust-repro-20260907`; it retains source, data, candidate-weight, evaluation, storage, artifact, execution, and boundary observations.
- The receipt records independently verified private data/model storage coordinates without copying private content into this repository record.

## Oracle legs

| Leg | Evidence |
|---|---|
| P1 | The receipt `data` block reports an exact original data match and identifies the Rust, balanced-data, and manifest SHA-256 values; the retained original receipt `benchmarks/model2vec_training/receipts/massed-20260906-a31a6b90.yml` identifies that evaluation data as crate-disjoint. |
| P2 | The receipt retains exact original matches and SHA-256 values only for the raw and post-SIF weight files. It contains no direct tokenizer hash/preservation gate or evidence that both candidates reload with finite 256-dimensional vectors; this observation therefore cannot decide P2. |
| P3 | The receipt `evaluation` block reports passed completion, exact dense metrics and real-repository aggregates, and confines the recorded variance to synthetic hybrid ties. |
| P4 | The receipt `storage`, `artifacts`, and `boundary` blocks retain retrievable artifact identities and state that the candidate remains preserved experiment evidence and is not deployed by the run. |
