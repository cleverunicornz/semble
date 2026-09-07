## Status

accepted

## Date

2026-09-07

## Context

This DELTA adds Model2Vec sample preparation, training, evaluation, and retained experiment receipts under `benchmarks/`. The fork needs a recorded boundary for the new work before its implementation and observations are interpreted as package behavior.

## Evidence

- `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_data/prepare.py` supplies the isolated sample-preparation command.
- `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/train_smoke.py`, `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/train_contrastive.py`, and `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/evaluate_full.py` supply the experiment paths.
- `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/README.md` declares the work a bounded quality signal rather than a production model build, explains the direct static-student path, and retains the full result as experiment evidence rather than a deployment candidate.
- `54cd1f1645cf0b73b39eecffe503341508e0cf80:benchmarks/model2vec_training/receipts/massed-20260907-1b2adaac.yml` records a reproduced experiment while stating that the candidate is not deployed.

## Decision

Keep the Model2Vec work Rust-first and benchmark-only. Use the direct static-student training path with explicit padding handling for its smoke and contrastive experiments; do not use Stock Tokenlearn or promote either resulting candidate into Semble's runtime configuration.

## Why

The added implementation is isolated to benchmark surfaces, and its retained evidence distinguishes successful mechanical reproduction from a production-quality decision. The direct path accommodates the selected static student's tokenizer and padding behavior without claiming that a training result changes the package's shipped model behavior.

## Rejected alternatives

- Treat the retained experiment receipts as production qualification or a runtime-model change — rejected because the added benchmark documentation and receipt delimit them as experiment evidence.
- Use Stock Tokenlearn for the selected smoke path — rejected because the added benchmark documentation identifies incompatible pooling and padding assumptions for these inputs.
- Enable the optional non-Rust sources by default — rejected because the added preparation surface makes Rust required and other sources explicit opt-ins.

## Consequences

The Model2Vec records remain scoped to `benchmarks/model2vec_data/` and `benchmarks/model2vec_training/`. Their residuals must exclude production integration, default-model selection, and fork gate assurance. The existing unassured fork-gate route remains unchanged.

## Revisit when

A separately recorded qualification establishes a bounded production-integration promise with its own oracle and evidence, or an evidence-backed adapter removes the documented incompatibility of the selected Tokenlearn path.