## State

implemented

## Judges

[P-000004](situation/promises/P-000004-full-rust-contrastive-evaluation.md)

## Inputs

- The pinned Rust train/evaluation files, manifest, replay sources, local baseline snapshots, and pinned repository receipt accepted by the full experiment scripts.
- The generated manifests, training receipts, evaluation receipt, summaries, and retained result files represented by `benchmarks/model2vec_training/receipts/massed-20260906-a31a6b90.yml` and `benchmarks/model2vec_training/receipts/massed-20260907-1b2adaac.yml`.
- The focused full-experiment helpers and scripts under `benchmarks/model2vec_training/`.

## Pass

- P1: The full preparation stage rejects mismatched Rust identities, preserves holdout separation, and produces the required balanced training manifest.
- P2: Full training writes reloadable raw and post-SIF candidates with finite 256-dimensional vectors and preserved tokenizer bytes.
- P3: The fixed four-model full-run invocation evaluates potion-v2, potion-v1, raw, and post-SIF candidates on both declared Rust retrieval surfaces and the complete pinned repository suite with paired comparisons.
- P4: Retained evidence identifies its source and artifact identities and explicitly keeps the resulting candidates as experiment evidence rather than a deployment decision.

## Fail

- F1: A mismatched Rust input, manifest, holdout overlap, or incomplete balanced training corpus proceeds as a full experiment.
- F2: A missing, non-finite, non-256-dimensional, or tokenizer-altering candidate is reported as a successful full training result.
- F3: A standalone `evaluate_full.py` receipt is treated as a full-experiment evaluation without the fixed four-model `run_cpu_full.sh` invocation, or an invalid evaluation surface or incomplete pinned repository suite is reported as a successful full evaluation.
- F4: The retained result lacks the provenance needed to identify its run or represents a candidate as selected or deployed.

## Implementation

`benchmarks/model2vec_training/run_cpu_full.sh` dispatches the full preparation, smoke qualification, training, repository materialization, evaluation, and archival route. `tests/benchmarks/test_model2vec_contrastive.py` and `tests/benchmarks/test_model2vec_full_evaluation.py` exercise focused executable seams.

## Implementation coverage

| Leg | Decision | Coverage |
|---|---|---|
| P1 | Validate the complete Rust inputs and construct balanced heldout-excluding replay data | `benchmarks/model2vec_training/prepare_contrastive.py::validate_rust_inputs`; `benchmarks/model2vec_training/prepare_contrastive.py::prepare`; `tests/benchmarks/test_model2vec_contrastive.py::test_replay_selection_applies_length_dedup_and_holdout` |
| P2 | Train and reload raw/post-SIF candidate artifacts with required mechanical gates | `benchmarks/model2vec_training/train_contrastive.py::train`; `tests/benchmarks/test_model2vec_contrastive.py::test_build_and_raw_export_preserve_tokenizer` |
| P3 | The fixed wrapper supplies all four comparison models; evaluation records both Rust surfaces, the repository suite, and paired comparisons | `benchmarks/model2vec_training/run_cpu_full.sh` (fixed four `--model` arguments); `benchmarks/model2vec_training/evaluate_full.py::evaluate`; `tests/benchmarks/test_model2vec_full_evaluation.py::test_dense_ranks_use_supplied_full_corpus_positions` |
| P4 | Retain provenance and the non-deployment boundary with the result | manual |
| F1 | Bad identity, holdout overlap, or wrong balanced corpus stops the preparation path | `benchmarks/model2vec_training/prepare_contrastive.py::validate_rust_inputs`; `benchmarks/model2vec_training/prepare_contrastive.py::prepare` |
| F2 | Failed full-training gates prevent a passing training receipt | `benchmarks/model2vec_training/train_contrastive.py::train` |
| F3 | Establish the fixed wrapper invocation manually; supplied model paths and false evaluation gates decide invalid input and surface cases | manual for the fixed-invocation boundary; `benchmarks/model2vec_training/evaluate_full.py::parse_model_specs`; `benchmarks/model2vec_training/evaluate_full.py::evaluate` |
| F4 | Missing provenance or a deployment interpretation fails the retained-evidence judgment | manual |
