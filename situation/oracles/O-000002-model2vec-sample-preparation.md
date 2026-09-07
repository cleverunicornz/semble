## State

implemented

## Judges

[P-000002](situation/promises/P-000002-model2vec-sample-preparation.md)

## Inputs

- Local JSONL fixtures and temporary output paths supplied by `tests/benchmarks/test_model2vec_adapters.py` and `tests/benchmarks/test_model2vec_prepare.py`.
- The preparation modules under `benchmarks/model2vec_data/` and the focused command documented in `benchmarks/model2vec_data/README.md`.

## Pass

- P1: A valid local Rust fixture produces all four canonical JSONL artifacts and a manifest whose listed artifact and input hashes match the written bytes.
- P2: Identical inputs and configuration produce byte-identical outputs; selected group identities do not cross the training/evaluation boundary, and held-out exact code/text is absent from all training-side outputs.
- P3: Structured dataset fields are accepted only as bounded mappings parsed without executing their strings; Swift test assertions are not treated as positive code.
- P4: Invalid configuration, missing required Rust input, and an occupied output destination are rejected without overwriting user data.

## Fail

- F1: A valid fixture lacks one required artifact or its manifest hash disagrees with the written bytes.
- F2: Deterministic selection, group separation, or evaluation-first exact-content exclusion is violated within the declared local-fixture scope.
- F3: A dataset string is executed, an invalid/oversized structured field is accepted as usable content, or a Swift assertion becomes positive code.
- F4: Invalid configuration, missing required Rust input, or an occupied output destination is accepted or overwritten.

## Implementation

`uv run --no-project --with pytest==9.0.3 python -m pytest -c /dev/null --rootdir=. --noconftest tests/benchmarks/test_model2vec_adapters.py tests/benchmarks/test_model2vec_prepare.py` is the executable local-fixture route.

## Implementation coverage

| Leg | Decision | Coverage |
|---|---|---|
| P1 | Valid fixture writes and hashes every declared artifact | `tests/benchmarks/test_model2vec_prepare.py::test_rust_only_run_writes_outputs_and_verified_manifest` |
| P2 | Deterministic selection and separated output content | `tests/benchmarks/test_model2vec_prepare.py::test_selection_is_deterministic_hash_mixed_and_seed_sensitive`; `tests/benchmarks/test_model2vec_prepare.py::test_crate_groups_never_cross_splits`; `tests/benchmarks/test_model2vec_prepare.py::test_exact_pair_and_cross_split_code_dedup` |
| P3 | Dataset fields stay data and Swift assertions stay out of positive code | `tests/benchmarks/test_model2vec_adapters.py::test_rust_never_executes_dataset_strings`; `tests/benchmarks/test_model2vec_adapters.py::test_swift_uses_translated_solution_never_ground_truth`; `tests/benchmarks/test_model2vec_adapters.py::test_optional_adapters_apply_field_size_bound` |
| P4 | Unsafe invocation fails without writing over user data | `tests/benchmarks/test_model2vec_prepare.py::test_cli_rejects_invalid_config_and_unsafe_output`; `tests/benchmarks/test_model2vec_prepare.py::test_prepare_sample_refuses_non_empty_output_dir` |
| F1 | Missing artifacts or incorrect written hashes fail the artifact assertions | `tests/benchmarks/test_model2vec_prepare.py::test_rust_only_run_writes_outputs_and_verified_manifest` |
| F2 | Cross-split group or content overlap fails the separation assertions | `tests/benchmarks/test_model2vec_prepare.py::test_crate_groups_never_cross_splits`; `tests/benchmarks/test_model2vec_prepare.py::test_corpus_text_dedup_including_kotlin_cross_split` |
| F3 | Executable/invalid/oversized input or assertion fallback fails the adapter assertions | `tests/benchmarks/test_model2vec_adapters.py::test_rust_never_executes_dataset_strings`; `tests/benchmarks/test_model2vec_adapters.py::test_swift_missing_solution_skips_even_when_ground_truth_exists`; `tests/benchmarks/test_model2vec_adapters.py::test_parse_serialized_mapping_accepts_only_mappings` |
| F4 | Accepted invalid invocation or overwrite fails the CLI/programmatic assertions | `tests/benchmarks/test_model2vec_prepare.py::test_resolve_inputs_missing_rust_is_config_error`; `tests/benchmarks/test_model2vec_prepare.py::test_check_output_dir_refuses_unsafe_destinations` |
