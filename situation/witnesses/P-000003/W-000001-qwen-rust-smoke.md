## Promise

[P-000003](situation/promises/P-000003-model2vec-training-smoke.md)

## Oracle

[O-000003](situation/oracles/O-000003-model2vec-training-smoke.md)

## Result

PASS

## Head

`ccd30f4f06bd2d954b676b2e68a07da11a99420a`

## Observed

2026-09-06

## Evidence

- `benchmarks/model2vec_training/receipts/massed-20260906-ccd30f4.yml` is the committed result file for H100 run `massed-20260906-ccd30f4`, including its source head, exact sample identities, pinned model revisions, gate observations, artifact digest, and boundary.
- The same receipt records successful remote and independent local CPU load-and-search verification after the runtime-dependency repair.

## Oracle legs

| Leg | Evidence |
|---|---|
| P1 | The receipt `inputs` block names the 894 training-pair and 100 evaluation-pair identities and the `models` block names both pinned snapshots. |
| P2 | The receipt `gates` and `metrics` blocks record a 256-dimensional candidate, zero PAD-vector norm, finite smoke values, and batch-invariance cosine above the oracle threshold. |
| P3 | The receipt `gates` block records `remote_cpu_load_and_search: passed-after-runtime-dependency-repair` and `independent_local_cpu_load_and_search: passed`. |
| P4 | The receipt `boundary` states that the 100-pair result is a smoke signal, that every reported retrieval metric regressed, and that the candidate is not a deployment candidate. |
