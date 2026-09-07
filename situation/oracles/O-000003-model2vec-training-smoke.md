## State

implemented

## Judges

[P-000003](situation/promises/P-000003-model2vec-training-smoke.md)

## Inputs

- The qualified pair files, manifest, and pinned local snapshots accepted by `benchmarks/model2vec_training/train_smoke.py`.
- The generated training and CPU-verification receipts, including `benchmarks/model2vec_training/receipts/massed-20260906-ccd30f4.yml`.
- The focused smoke helpers and runner scripts under `benchmarks/model2vec_training/`.

## Pass

- P1: The runner accepts only the qualified sample identity and concrete local model snapshots before teacher loading.
- P2: The exported candidate reloads at 256 dimensions with finite vectors, an exactly zero PAD vector, finite training/teacher values, and minimum batch-invariance cosine at least `0.99999`.
- P3: CPU verification runs with CUDA hidden, loads the candidate through Semble, yields finite embedded vectors and search results, and writes its receipt.
- P4: The retained result labels its held-out metrics as smoke evidence rather than a production qualification.

## Fail

- F1: A mismatched qualified sample or incomplete local snapshot proceeds into training.
- F2: Any candidate export gate in P2 is false but the training receipt reports a pass.
- F3: Any CPU-verification gate in P3 is false but the verification receipt reports a pass.
- F4: The retained smoke result promotes or deploys its candidate, or represents its held-out metric as a production qualification.

## Implementation

`benchmarks/model2vec_training/run_gpu_smoke.sh` dispatches the GPU training and its CPU verifier; `tests/benchmarks/test_model2vec_training_core.py` and `tests/benchmarks/test_model2vec_training_smoke.py` exercise their local executable seams.

## Implementation coverage

| Leg | Decision | Coverage |
|---|---|---|
| P1 | Exact sample identity and snapshots are required before training | `benchmarks/model2vec_training/train_smoke.py::_validate_sample_manifest`; `tests/benchmarks/test_model2vec_training_smoke.py::test_full_smoke_and_cpu_verification` |
| P2 | Exported candidate satisfies every training gate | `benchmarks/model2vec_training/train_smoke.py::run`; `tests/benchmarks/test_model2vec_training_core.py::test_training_export_zeros_pad_and_is_batch_invariant` |
| P3 | CPU verifier loads, embeds, and searches the candidate | `benchmarks/model2vec_training/verify_smoke.py::run`; `tests/benchmarks/test_model2vec_training_smoke.py::test_full_smoke_and_cpu_verification` |
| P4 | Receipt retains the smoke-only quality boundary | manual |
| F1 | Input/snapshot mismatch stops the smoke path | `benchmarks/model2vec_training/train_smoke.py::_validate_sample_manifest`; `benchmarks/model2vec_training/train_smoke.py::_validate_local_snapshot` |
| F2 | Any false training gate produces a failed training receipt | `benchmarks/model2vec_training/train_smoke.py::run` |
| F3 | Any false verification gate produces a failed verification receipt | `benchmarks/model2vec_training/verify_smoke.py::run` |
| F4 | A production interpretation is rejected from the retained evidence | manual |
