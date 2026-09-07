## State

implemented

## Promise

For valid local Rust input and a new or empty destination, the Model2Vec sample-preparation surface writes canonical pair and corpus training/evaluation JSONL artifacts plus a manifest; identical input and configuration select the same artifacts, held-out evaluation content is excluded from training-side exact code/text, and dataset field strings are treated as data rather than executable input.

## Scope

This promise covers the local JSONL path through `benchmarks/model2vec_data/prepare.py`, its adapters, and `prepare_sample()` for the configured Rust source and explicitly enabled optional sources. It includes input/configuration validation, output-destination protection, deterministic selection and group routing over scanned rows, exact-content deduplication across the four output artifacts, and their manifest hashes. It covers only the preparation process and supplied local files.

## Oracle

[O-000002](situation/oracles/O-000002-model2vec-sample-preparation.md)

## State evidence

- `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_data/prepare.py` adds the implementation in this closure's declared delta.
- `42fc95450b74b2f96e37c931dc16b5441319cafd:tests/benchmarks/test_model2vec_adapters.py` and `42fc95450b74b2f96e37c931dc16b5441319cafd:tests/benchmarks/test_model2vec_prepare.py` add executable coverage for the local preparation behavior.

## Residual

This promise does not attest that a local file matches a pinned Hub revision, assure Parquet behavior, assert training sufficiency or retrieval quality, run model training, or select or deploy a model.

## References

- `benchmarks/model2vec_data/README.md`
- [D-000002](situation/decisions/D-000002-safe-model2vec-data-interpretation.md)
- [I-000001](situation/invariants/I-000001-model2vec-data-is-never-executed.md)
