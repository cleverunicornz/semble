## State

open

## Gap

The synthetic full-corpus hybrid comparison has no recorded deterministic total order for equal-score, equal-start-line candidates.

## Relevance

[P-000003](situation/promises/P-000003-model2vec-training-experiment.md) includes full comparison evidence. A bounded non-deterministic tie surface weakens exact reproducibility of that experiment without establishing a change in the retained quality conclusion.

## Evidence

- `benchmarks/model2vec_training/receipts/massed-20260907-1b2adaac.yml` records that its differing summary values are confined to synthetic hybrid evaluation, attributes them to process-random set order for equal start-line and score ties, and states that the conclusion did not change.
- `benchmarks/model2vec_training/README.md` retains the same boundary: synthetic pair-corpus hybrid metrics vary because the historical evaluator did not use a total deterministic candidate key or pin `PYTHONHASHSEED`.

## Impact

Exact synthetic-hybrid summary values cannot yet be represented as bitwise-reproducible evidence. The recorded result remains bounded experiment evidence; this gap does not imply a failure on real-repository or dense evaluation surfaces.

## Resolution

none

## References

- `benchmarks/model2vec_training/receipts/massed-20260907-1b2adaac.yml`
- [P-000003](situation/promises/P-000003-model2vec-training-experiment.md)