## State

implemented

## Promise

Given the qualified smoke sample and pinned local teacher and student snapshots, the Model2Vec smoke surface trains and exports a standard 256-dimensional candidate with a zero PAD vector, records its finite retrieval and training gates, and CPU-loads and searches the export through Semble; its held-out metrics remain smoke evidence rather than a deployment claim.

## Scope

This promise covers `benchmarks/model2vec_training/train_smoke.py`, `benchmarks/model2vec_training/verify_smoke.py`, and `benchmarks/model2vec_training/run_gpu_smoke.sh` for the qualified 894-pair training and 100-pair evaluation sample, the pinned Qwen3 teacher and potion v2 student, their generated receipt files, and the CPU verification path. It covers only this experimental runner and its supplied local snapshots.

## Oracle

[O-000003](situation/oracles/O-000003-model2vec-training-smoke.md)

## State evidence

- `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_training/train_smoke.py` and `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_training/verify_smoke.py` add the implementation in this closure's declared delta.
- [W-000001](situation/witnesses/P-000003/W-000001-qwen-rust-smoke.md) retains one completed H100 observation of the bounded runner.

## Residual

This promise does not assure a quality improvement, production retrieval behavior, a current-head runnable gate, availability of the external GPU environment, or promotion or deployment of the candidate.

## References

- `benchmarks/model2vec_training/README.md`
- [D-000003](situation/decisions/D-000003-retain-model2vec-results-as-experiment-evidence.md)
- [I-000002](situation/invariants/I-000002-model2vec-results-require-recorded-promotion.md)
