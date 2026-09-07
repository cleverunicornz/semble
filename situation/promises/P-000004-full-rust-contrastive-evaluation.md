## State

implemented

## Promise

The full Rust contrastive experiment verifies the prescribed input identities, trains raw and post-SIF static Model2Vec candidates from the balanced Rust-plus-replay corpus, evaluates them on the held-out Rust and pinned-repository surfaces, and retains the resulting comparison evidence without selecting or deploying either candidate.

## Scope

This promise covers `benchmarks/model2vec_training/prepare_contrastive.py`, `train_contrastive.py`, `evaluate_full.py`, `sync_evaluation_repos.py`, and `run_cpu_full.sh`; the pinned model, Rust, replay, and repository inputs they name; both candidate artifact directories; and their emitted manifests, receipts, and summaries. It covers only the full offline experiment and supplied pinned inputs.

## Oracle

[O-000004](situation/oracles/O-000004-full-rust-contrastive-evaluation.md)

## State evidence

- `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_training/prepare_contrastive.py`, `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_training/train_contrastive.py`, and `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_training/evaluate_full.py` add the implemented experimental stages in this closure's declared delta.
- [W-000002](situation/witnesses/P-000004/W-000002-full-rust-contrastive-run.md) and [W-000003](situation/witnesses/P-000004/W-000003-full-rust-contrastive-reproduction.md) retain completed observations of the full run and its reproduction.

## Residual

This promise does not assure an improvement over any baseline, production retrieval behavior, a current-head execution route, availability of external CPU, datasets, models, or private artifacts, or promotion or deployment of a candidate.

## References

- `benchmarks/model2vec_training/README.md`
- [D-000003](situation/decisions/D-000003-retain-model2vec-results-as-experiment-evidence.md)
- [I-000002](situation/invariants/I-000002-model2vec-results-require-recorded-promotion.md)
- [G-000003](situation/gaps/G-000003-synthetic-hybrid-tie-order-variance.md)
