## Status

accepted

## Date

2026-09-07

## Context

The new Model2Vec preparation surface consumes dataset fields that can contain serialized mappings and, for Swift, serialized test assertions. The preparation path needs normalized query/code or corpus data without granting those strings execution authority or substituting test material for missing positive code.

## Evidence

- `42fc95450b74b2f96e37c931dc16b5441319cafd:benchmarks/model2vec_data/adapters.py` bounds structured fields before parsing, accepts mappings through `json.loads` and `ast.literal_eval`, and keeps Rust `test` and Swift `ground_truth` out of positive code.
- `42fc95450b74b2f96e37c931dc16b5441319cafd:tests/benchmarks/test_model2vec_adapters.py` supplies malicious-literal, oversized-field, and Swift-fallback cases.
- [P-000002](situation/promises/P-000002-model2vec-sample-preparation.md) and [O-000002](situation/oracles/O-000002-model2vec-sample-preparation.md) bound the resulting repository behavior and judgment route.

## Decision

Interpret Model2Vec dataset strings only as bounded data mappings, never as executable content, and require an explicit positive-code field rather than falling back to serialized test assertions.

## Why

The selected boundary preserves the source's query/code semantics while preventing untrusted data from executing in the preparation process and preventing test assertions from being mislabeled as training positives.

## Rejected alternatives

- Execute or evaluate dataset-provided strings or test assertions. Rejected because dataset content is not trusted program authority.
- Use Swift `ground_truth` or translated test cases when `translated_solution` is absent. Rejected because assertions are metadata, not verified positive code, and would change the sample's meaning.

## Consequences

- [I-000001](situation/invariants/I-000001-model2vec-data-is-never-executed.md) makes the data-only boundary binding.
- [P-000002](situation/promises/P-000002-model2vec-sample-preparation.md) retains the behavior; [O-000002](situation/oracles/O-000002-model2vec-sample-preparation.md) names its executable checks.

## Revisit when

An audited, non-executing parser and a verified source-schema change provide evidence that the positive-code boundary should be replaced.
