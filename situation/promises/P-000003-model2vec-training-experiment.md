## State

implemented

## Promise

With valid explicitly supplied prepared data, local model snapshots, and the route's required execution environment, the Model2Vec training benchmark provides two bounded routes. The GPU smoke route checks its prepared sample, trains an exported static candidate, and invokes CPU load, encode, and search verification. The CPU full route exports raw and post-SIF candidates and emits full comparison evidence only for `potion-v1`, `potion-v2`, `candidate-raw`, and `candidate-post-sif`, with `potion-v2` as the baseline.

## Scope

This promise covers the Model2Vec benchmark implementation under `benchmarks/model2vec_training/`, its focused tests under `tests/benchmarks/test_model2vec_contrastive.py`, `tests/benchmarks/test_model2vec_full_evaluation.py`, `tests/benchmarks/test_model2vec_training_core.py`, and `tests/benchmarks/test_model2vec_training_smoke.py`, and artifacts written to caller-selected benchmark output directories. It applies only to the supplied benchmark inputs and execution environments, not to the package's runtime model selection.

## Oracle

[O-000003](situation/oracles/O-000003-model2vec-training-experiment.md)

## State evidence

- `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/run_gpu_smoke.sh`, `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/run_cpu_full.sh`, `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/train_smoke.py`, `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/train_contrastive.py`, and `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/evaluate_full.py` are the implementation introduced by the trigger head.
- [D-000002](situation/decisions/D-000002-model2vec-benchmark-boundary.md) records the benchmark-only direct-static-student decision.

## Residual

This promise does not assure public or private input availability, a completed remote run, any model-quality improvement, the numerical results in retained receipts, production integration or default-model selection, or a fork gate claim.

## References

- `benchmarks/model2vec_training/README.md`
- `benchmarks/model2vec_training/receipts/massed-20260906-a31a6b90.yml`
- `benchmarks/model2vec_training/receipts/massed-20260907-1b2adaac.yml`