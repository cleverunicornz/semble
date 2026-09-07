## State

implemented

## Judges

[P-000002](situation/promises/P-000002-model2vec-sample-preparation.md)

## Inputs

- The preparation implementation under `benchmarks/model2vec_data/`.
- The focused executable tests `tests/benchmarks/test_model2vec_adapters.py` and `tests/benchmarks/test_model2vec_prepare.py` with their local JSONL fixtures.
- Valid and invalid source selections and output destinations within the promise Scope.

## Pass

The promise passes only when all of these legs hold within its Scope:

1. A valid Rust input and safe destination produce the normalized output files and a manifest that records their hashes.
2. Identical selected inputs and configuration produce byte-identical output files.
3. Evaluation-owned exact code or text is absent from the corresponding training outputs, available crate or repository groups do not cross the split boundary, and the Swift grouping limitation is visible in the manifest.
4. Optional TypeScript, Swift, and Kotlin inputs require explicit enablement while Rust remains required.
5. Unsafe destinations are refused, and dataset strings are accepted only as safely parsed mappings without execution.

## Fail

The promise fails when any applicable leg is contradicted within its Scope:

1. A valid Rust input and safe destination omit normalized output or its manifest hashes.
2. Identical selected inputs and configuration yield different output bytes.
3. Evaluation-owned exact code or text remains in the corresponding training output, an available crate or repository group crosses the split boundary, or the Swift grouping limitation is hidden from the manifest.
4. An optional source contributes without explicit enablement or a missing Rust input is accepted as a successful preparation.
5. An unsafe destination is accepted, or a dataset string outside the safe mapping grammar is executed or accepted as a usable mapping.

## Implementation

The focused command is recorded in `benchmarks/model2vec_data/README.md`:

```bash
uv run --no-project --with pytest==9.0.3 python -m pytest -c /dev/null --rootdir=. --noconftest \
  tests/benchmarks/test_model2vec_adapters.py tests/benchmarks/test_model2vec_prepare.py
```

## Implementation coverage

| Leg | Decision | Coverage |
|---|---|---|
| P1 | Normalized output and manifest hashes are emitted | `tests/benchmarks/test_model2vec_prepare.py::test_rust_only_run_writes_outputs_and_verified_manifest` |
| P2 | Output bytes are stable for identical inputs and configuration | `tests/benchmarks/test_model2vec_prepare.py::test_selection_is_deterministic_hash_mixed_and_seed_sensitive` |
| P3 | Split and exact-content boundaries are retained and Swift grouping is disclosed | `tests/benchmarks/test_model2vec_prepare.py::test_crate_groups_never_cross_splits`; `tests/benchmarks/test_model2vec_prepare.py::test_exact_pair_and_cross_split_code_dedup`; `tests/benchmarks/test_model2vec_prepare.py::test_swift_weak_grouping_visible_in_manifest` |
| P4 | Optional-source activation and required Rust input are decided | `tests/benchmarks/test_model2vec_prepare.py::test_optional_typescript_shortfall_does_not_block_rust`; `tests/benchmarks/test_model2vec_prepare.py::test_resolve_inputs_missing_rust_is_config_error` |
| P5 | Safe parser and destination checks are decided | `tests/benchmarks/test_model2vec_adapters.py::test_rust_never_executes_dataset_strings`; `tests/benchmarks/test_model2vec_prepare.py::test_check_output_dir_refuses_unsafe_destinations` |
| F1 | Omitted normalized output or manifest hashes are detected | `tests/benchmarks/test_model2vec_prepare.py::test_rust_only_run_writes_outputs_and_verified_manifest` |
| F2 | Output-byte instability is detected | `tests/benchmarks/test_model2vec_prepare.py::test_selection_is_deterministic_hash_mixed_and_seed_sensitive` |
| F3 | Split, exact-content, and Swift-disclosure contradictions are detected | `tests/benchmarks/test_model2vec_prepare.py::test_crate_groups_never_cross_splits`; `tests/benchmarks/test_model2vec_prepare.py::test_exact_pair_and_cross_split_code_dedup`; `tests/benchmarks/test_model2vec_prepare.py::test_swift_weak_grouping_visible_in_manifest` |
| F4 | Implicit optional-source contribution and missing Rust input are detected | `tests/benchmarks/test_model2vec_prepare.py::test_optional_typescript_shortfall_does_not_block_rust`; `tests/benchmarks/test_model2vec_prepare.py::test_resolve_inputs_missing_rust_is_config_error` |
| F5 | Unsafe destinations and unsafe mapping handling are detected | `tests/benchmarks/test_model2vec_adapters.py::test_rust_never_executes_dataset_strings`; `tests/benchmarks/test_model2vec_prepare.py::test_check_output_dir_refuses_unsafe_destinations` |