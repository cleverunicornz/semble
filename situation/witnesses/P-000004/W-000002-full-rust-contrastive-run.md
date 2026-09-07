## Promise

[P-000004](situation/promises/P-000004-full-rust-contrastive-evaluation.md)

## Oracle

[O-000004](situation/oracles/O-000004-full-rust-contrastive-evaluation.md)

## Result

PASS

## Head

`f586c664ae121da213910daabf7d8b3dd1ea81b2`

## Observed

2026-09-06

## Evidence

- `benchmarks/model2vec_training/receipts/massed-20260906-a31a6b90.yml` is the committed result file for full CPU run `mc-a31a6b90-20260906`; it retains the source head, input and artifact SHA-256 values, training/evaluation observations, environment, completion gates, and non-deployment boundary.
- The receipt's `artifacts` block retains the final archive digest, and its `execution` block records that the disposable instance was terminated.

## Oracle legs

| Leg | Evidence |
|---|---|
| P1 | The receipt `inputs` block identifies the pinned Rust source, declares the evaluation split crate-disjoint, and names the training/evaluation identities and balanced training corpus; `gates` records `full_training_rows: passed`. |
| P2 | The receipt `models` and `training` blocks identify both candidate artifacts and their dimensions; `gates` records finite candidate loading, vector, and tokenizer checks. |
| P3 | The receipt `evaluation` block contains the held-out/full-corpus Rust and real-repository observations; `gates` records complete held-out Rust, full Rust corpus, and repository-suite evaluation. |
| P4 | The receipt `source`, `artifacts`, and `boundary` blocks identify retained evidence and state that the candidate is promising experiment evidence rather than a promoted or deployed candidate. |
