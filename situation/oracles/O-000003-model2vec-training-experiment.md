## State

implemented

## Judges

[P-000003](situation/promises/P-000003-model2vec-training-experiment.md)

## Inputs

- The training implementation under `benchmarks/model2vec_training/`, including `run_gpu_smoke.sh`, `run_cpu_full.sh`, `train_smoke.py`, `verify_smoke.py`, `train_contrastive.py`, and `evaluate_full.py`.
- The focused tests `tests/benchmarks/test_model2vec_training_core.py`, `tests/benchmarks/test_model2vec_training_smoke.py`, `tests/benchmarks/test_model2vec_contrastive.py`, and `tests/benchmarks/test_model2vec_full_evaluation.py`.
- Valid and malformed explicitly supplied benchmark inputs, model paths, output directories, and comparison labels within the promise Scope.
- The emitted CPU full-route `train.jsonl`, line-aligned `provenance.jsonl`, and `manifest.json` produced by `prepare_contrastive.py`.

## Pass

The promise passes only when all of these legs hold within its Scope:

1. The GPU route requires the prepared sample inputs, invokes smoke training, and invokes CPU verification with CUDA hidden.
2. The CPU full route validates its complete pinned Rust inputs, excludes held-out Rust content while constructing replay data, and materializes each benchmark repository at its manifest revision.
3. The CPU full route emits attested balanced training data: `manifest.json` records Rust input revision and digests, each replay source's dataset, revision, and selected count, and the SHA-256 and row count of both `train.jsonl` and `provenance.jsonl`; provenance records exactly Rust plus the six replay languages with `per_language` rows apiece, and `total_pairs` and both output row counts equal their sum.
4. The CPU full route invokes contrastive training to create raw and post-SIF candidate artifacts from its supplied base model.
5. Full comparison accepts exactly the fixed four model labels and `potion-v2` baseline before it emits comparison evidence.

## Fail

The promise fails when any applicable leg is contradicted within its Scope:

1. The GPU route omits a required prepared sample input, bypasses smoke training, or bypasses CPU verification with CUDA hidden.
2. The CPU full route accepts incomplete or unpinned Rust inputs, retains held-out Rust content in replay, or accepts a benchmark checkout other than its manifest revision.
3. The CPU full route omits required training-data attestations, reports a missing, extra, or unequal per-language count, or has a `total_pairs` or output row count that disagrees with their required sum.
4. The CPU full route fails to invoke contrastive training for the raw and post-SIF candidates from its supplied base model.
5. Full comparison accepts an incomplete or shifted label set, or a baseline other than `potion-v2`.

## Implementation

`benchmarks/model2vec_training/run_gpu_smoke.sh` implements the GPU sequence, and `benchmarks/model2vec_training/run_cpu_full.sh` implements the CPU full route. The latter builds its isolated environment from `benchmarks/model2vec_training/requirements-cpu.lock` and executes focused contrastive and full-evaluation checks. The remaining focused checks are executable in that environment with:

```bash
"$VENV/bin/python" -m pytest -c /dev/null --rootdir=. --noconftest \
  tests/benchmarks/test_model2vec_training_core.py \
  tests/benchmarks/test_model2vec_training_smoke.py \
  tests/benchmarks/test_model2vec_contrastive.py \
  tests/benchmarks/test_model2vec_full_evaluation.py
```

## Implementation coverage

| Leg | Decision | Coverage |
|---|---|---|
| P1 | GPU preparation, smoke, and CPU-verification sequence is invoked | `benchmarks/model2vec_training/run_gpu_smoke.sh`; `tests/benchmarks/test_model2vec_training_smoke.py::test_full_smoke_and_cpu_verification` |
| P2 | Pinned Rust inputs, held-out exclusion, and exact repository materialization are decided | `benchmarks/model2vec_training/prepare_contrastive.py::validate_rust_inputs`; `benchmarks/model2vec_training/sync_evaluation_repos.py::sync_one`; `tests/benchmarks/test_model2vec_contrastive.py::test_replay_selection_applies_length_dedup_and_holdout` |
| P3 | Complete training-data attestation and per-language balance are decided | manual |
| P4 | CPU full route invokes contrastive training for both candidate outputs | `benchmarks/model2vec_training/run_cpu_full.sh`; `tests/benchmarks/test_model2vec_contrastive.py::test_build_and_raw_export_preserve_tokenizer` |
| P5 | Exact full-comparison labels and baseline are required | `benchmarks/model2vec_training/evaluate_full.py::validate_comparison_specs`; `tests/benchmarks/test_model2vec_full_evaluation.py::test_comparison_requires_all_four_models_and_v2_baseline` |
| F1 | Missing inputs or a bypassed GPU sequence terminate the route | `benchmarks/model2vec_training/run_gpu_smoke.sh`; `tests/benchmarks/test_model2vec_training_smoke.py::test_full_smoke_and_cpu_verification` |
| F2 | Unpinned or incomplete Rust inputs, held-out replay overlap, and wrong checkout revisions are rejected | `benchmarks/model2vec_training/prepare_contrastive.py::validate_rust_inputs`; `benchmarks/model2vec_training/sync_evaluation_repos.py::sync_one`; `tests/benchmarks/test_model2vec_contrastive.py::test_replay_selection_applies_length_dedup_and_holdout` |
| F3 | Missing or mismatched complete training-data attestation and balance are detected | manual |
| F4 | Missing contrastive candidate generation terminates the CPU route | `benchmarks/model2vec_training/run_cpu_full.sh`; `benchmarks/model2vec_training/train_contrastive.py::train` |
| F5 | Incomplete, shifted, or wrong-baseline comparisons are rejected | `benchmarks/model2vec_training/evaluate_full.py::validate_comparison_specs`; `tests/benchmarks/test_model2vec_full_evaluation.py::test_comparison_requires_all_four_models_and_v2_baseline` |